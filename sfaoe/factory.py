"""集中构建模型，避免训练入口混入多个实验模型的初始化细节。"""


def build_model(args, nb_filter, num_blocks):
    """返回未初始化、未迁移到 GPU 的模型；初始化顺序由训练器控制。"""
    from model.model_DNANet import Res_CBAM_block

    if args.model == "DNANet":
        from sfaoe.backbone import DNANet

        return DNANet(
            num_classes=1,
            input_channels=args.in_channels,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            moe_stages=args.moe_stages,
            dilations=args.dilations,
            noise_scale=args.noise_scale,
            patch_size=args.patch_size,
            top_k=args.top_k,
            input_routing=args.input_routing,
        )
    if args.model == "s4decode":
        from model.model_decode import DNANet

        return DNANet(
            num_classes=1,
            input_channels=args.in_channels,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
        )
    if args.model == "MambaIR":
        from model.mambairv2light import MambaIRv2Light

        return MambaIRv2Light(
            batch_size=args.train_batch_size,
            upscale=1,
            img_size=256,
            embed_dim=48,
            d_state=8,
            depths=[2, 3, 3, 2],
            num_heads=[4, 4, 4, 4],
            window_size=16,
            inner_rank=32,
            num_tokens=64,
            convffn_kernel_size=5,
            img_range=1.0,
            mlp_ratio=1.0,
            upsampler="pixelshuffledirect",
        )
    raise ValueError(f"Unsupported model: {args.model!r}")
