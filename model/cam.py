# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model.mix_blocks import ShuffleViTBlock, Res_CBAM_block, ShuffleV2Block, Freq_Res_CBAM_block, Freq_Shuffle_Block
import torch
import torch.nn as nn
from torchvision      import transforms


class SPG_block(nn.Module):
    def __init__(self, inp_channel_list, oup_channel):
        super(SPG_block, self).__init__()
        inp_channel = inp_channel_list[0]
        inp_channel_sum = sum(inp_channel_list)
        self.convup = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
            # nn.Conv2d(inp_channel//2, inp_channel//2, kernel_size=3, padding=(1,1), stride=2),
            # nn.BatchNorm2d(inp_channel//2),
            # nn.ReLU()
        )
        self.convdown = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(inp_channel, inp_channel, kernel_size=1, stride=1),
            # nn.BatchNorm2d(inp_channel),
            # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp_channel_sum, oup_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(oup_channel),
            nn.ReLU()
        )

    def forward(self, input):
        # x_in, norm_1, flops = input
        x0, down_layer, up_layer = input
        if down_layer is not None:
            x2 = self.convdown(down_layer)
            x0 = torch.cat([x0, x2], 1)
        if up_layer is not None:
            x1 = self.convup(up_layer)
            x0 = torch.cat([x0, x1], 1)
        x = self.conv1(x0)
        return x

    def get_flops(self):
        mskc_flops = self.mask_c.get_flops()
        return mskc_flops, self.norm_c


class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False, learn_loss_mask=False):   # [16, 32, 64, 128, 256] [2,2,2,2]
        super(DNANet, self).__init__()
        self.learn_loss_mask = learn_loss_mask
        self.block = block
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, [input_channels], nb_filter[0])
        self.conv1_0 = self._make_layer(block, [nb_filter[0]],  nb_filter[1], num_blocks[0], stride=2)
        self.conv2_0 = self._make_layer(block, [nb_filter[1]],  nb_filter[2], num_blocks[1], stride=2)
        self.conv3_0 = self._make_layer(block, [nb_filter[2]],  nb_filter[3], num_blocks[2], stride=2)
        self.conv4_0 = self._make_layer(block, [nb_filter[3]],  nb_filter[4], num_blocks[3], stride=2)

        self.conv0_1 = self._make_layer(block, [nb_filter[0], nb_filter[1]],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, [nb_filter[1], nb_filter[2], nb_filter[0]],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, [nb_filter[2], nb_filter[3], nb_filter[1]],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, [nb_filter[3], nb_filter[4], nb_filter[2]],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, [nb_filter[0], nb_filter[0], nb_filter[1]], nb_filter[0])
        self.conv1_2 = self._make_layer(block, [nb_filter[1], nb_filter[1], nb_filter[2], nb_filter[0]], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, [nb_filter[2], nb_filter[2], nb_filter[3], nb_filter[1]], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, [nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]], nb_filter[0])
        self.conv1_3 = self._make_layer(block, [nb_filter[1], nb_filter[1], nb_filter[1], nb_filter[2], nb_filter[0]], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, [nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, [nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[0]], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

        if self.learn_loss_mask:
            self.lossconv1x1 = nn.Conv2d (nb_filter[0]*2, 1, kernel_size=1)
            self.loss_spattn = nn.Conv2d (2, 1, kernel_size=1)

    def _make_layer(self, block, input_ch_list, output_channels, num_blocks=1,stride=1):
        layers = []
        input_channels = sum(input_ch_list)
        if input_channels > 4:
            layers.append(SPG_block(input_ch_list, input_channels//4))
            input_channels = input_channels//4
        layers.append(block(input_channels, output_channels, stride))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)  #  in (b 3 256 256)  x0_0 (b 16 256 256)

        x1_0 = self.conv1_0((x0_0, None, None))  # x1_0(b 32 128 128)
        x0_1 = self.conv0_1((x0_0, x1_0, None))    # x0_1 (b 16 256 256)

        x2_0 = self.conv2_0((x1_0, None, None))    # x2_0 (b 64 64 64)
        x1_1 = self.conv1_1((x1_0, x2_0, x0_1))    # x1_1 (b 32 128 128)
        x0_2 = self.conv0_2((torch.cat([x0_0, x0_1], 1), x1_1, None))  # x0_2 (b 16 256 256)

        x3_0 = self.conv3_0((x2_0, None, None))    # x3_0 (b 128 32 32)
        x2_1 = self.conv2_1((x2_0, x3_0, x1_1))    # x2_1 (b 64 64 64)
        x1_2 = self.conv1_2((torch.cat([x1_0, x1_1], 1), x2_1, x0_2))  # x1_2 (b 32 128 128)
        x0_3 = self.conv0_3((torch.cat([x0_0, x0_1, x0_2], 1), x1_2, None)) # x0_3 (b 16 256 256)

        x4_0 = self.conv4_0((x3_0, None, None))    # x4_0 (b 256 16 16)
        x3_1 = self.conv3_1((x3_0, x4_0, x2_1))
        x2_2 = self.conv2_2((torch.cat([x2_0, x2_1], 1), x3_1, x1_2))
        x1_3 = self.conv1_3((torch.cat([x1_0, x1_1, x1_2], 1), x2_2, x0_3))    # x1_3 (b 32 128 128)
        x0_4 = self.conv0_4((torch.cat([x0_0, x0_1, x0_2, x0_3], 1), x1_3, None)) # x0_4 (b 16 256 256)

        Final_x0_4 = self.conv0_4_final((
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1), None, None)) # Final_x0_4 (b 16 256 256)

        # flops

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [x4_0, x3_1, x2_2, x1_3, x0_4, output1, output2, output3, output4] # , maskconv
        else:
            output = self.final(Final_x0_4)
            return output   # , maskconv


def get_grad_heatmap(features, pred_class, visual_heatmap=True):
    """
    features = model.features(img)
    output = model.classifier(features)
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    """
    def extract(g):
        global features_grad
        features_grad = g

    # pred = torch.argmax(output).item()
    # pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()

    grads = features_grad

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    pooled_grads = pooled_grads[0]
    features = features[0]

    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()


def draw_CAM(model, img_path, save_dir, transform=None, visual_heatmap=False, mode=None):

    img = Image.open(img_path).convert('RGB')
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    img = input_transform(img)
    # if transform:
    #     img = transform(img)
    img = img.unsqueeze(0)

    model.eval()

    features = model(img)
    if mode == 'DNA':
        x4_0, x3_1, x2_2, x1_3, x0_4, output1, output2, output3, output4 = features
        feat_name=['x4_0', 'x3_1', 'x2_2', 'x1_3', 'x0_4', 'output1', 'output2', 'output3', 'output4']
    elif mode == 'Unet':
        x4_0, x3_1, x2_2, x1_3, x0_4, Final_x0_4, output = features
        feat_name = ['x4_0', 'x3_1', 'x2_2', 'x1_3', 'x0_4', 'Final_x0_4', 'output']
    def extract(g):
        global features_grad
        features_grad = g

    img = cv2.imread(img_path)

    for feat, name in zip(features, feat_name):
        feat = feat.squeeze()
        heatmap = feat.detach().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        if visual_heatmap:
            plt.matshow(heatmap)
            plt.show()

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        save_path = save_dir + name+'.png'
        cv2.imwrite(save_path, superimposed_img)


def for_CAM(img_path, check_dir, model, mode=None):
    if not os.path.exists('../cam_results/%s'%check_dir):
        os.mkdir('../cam_results/%s'%check_dir)
    save_path = ('../cam_results/%s/'%check_dir)+img_path.split('/')[-1][:-4]+'_'
    draw_CAM(model, img_path, save_path, mode=mode)

def for_inference(img_path, check_dir, model):
    img = Image.open(img_path).convert('RGB')
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    img = input_transform(img)
    img = img.unsqueeze(0)
    model.eval()
    output = model(img)[-1]
    if not os.path.exists('../visual_results/%s'%check_dir):
        os.mkdir('../visual_results/%s'%check_dir)
    save_dir = ('../visual_results/%s/'%check_dir)+img_path.split('/')[-1][:-4]+'_'
    save_path = save_dir + '.png'
    output = output.sigmoid()>0.1
    output = output.squeeze().detach().numpy()
    output = np.uint8(255 * output)
    cv2.imwrite(save_path, output)

if __name__ == '__main__':
    mode = 'FANet'
    img_path = '../dataset/IRSTD/images/XDU997.png'   # 512 512
    if mode == 'DNA':
        check_dir = 'IRSTD_UNet_05_06_2024_22_27_36_wDS' #'IRSTD_DNANet_07_04_2024_20_12_45_wDS' #
        from model.model_DNANet import DNANet
        model = DNANet(num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks=[2,2,2,2], nb_filter=[16,32,64,128,256], deep_supervision=True, cam=True)
        checkpoint_path = '../result/%s/mIoU__DNANet_IRSTD_epoch.pth.tar'%check_dir
    elif mode == 'Unet':
        check_dir = 'IRSTD_UNet_05_07_2024_21_41_36_wDS' #'IRSTD_UNet_07_06_2024_16_11_03_wDS' #'IRSTD_UNet_25_04_2024_16_34_08_wDS'
        from model.model_Unet import UNet
        model = UNet(num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks=[2,2,2,2], nb_filter=[16,32,64,128,256], deep_supervision=False, cam=True)
        checkpoint_path = '../result/%s/mIoU__UNet_IRSTD_epoch.pth.tar'%check_dir
    elif mode == 'FANet':
        check_dir = 'IRSTD_FANet_28_07_2024_16_52_04_wDS' #'IRSTD_UNet_25_04_2024_16_34_08_wDS'
        from model.model_FANet import FANet
        model = FANet(num_classes=1,input_channels=3, block=Freq_Shuffle_Block, num_blocks=[2,2,2,2], nb_filter=[16,32,64,128,256], deep_supervision=False, cam=True)
        checkpoint_path = '../result/%s/mIoU__FANet_IRSTD_epoch.pth.tar'%check_dir

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    for_CAM(img_path, check_dir, model, mode=mode)
    # for_inference(img_path, check_dir, model)

# import matplotlib.pyplot as plt
# feat = feat.squeeze()
# heatmap = feat.detach().numpy()
# heatmap = np.mean(heatmap, axis=0)
#
# heatmap = np.maximum(heatmap, 0)
# heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
# plt.show()