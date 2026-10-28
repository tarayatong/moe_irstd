# Context sparse branch validation

Validated on 2026-10-28 using Python 3.12 / PyTorch 2.8, CPU only.

- `python -m unittest discover -s tests -v`: 14 tests passed.
- Default full backbone has 15 MoEs, each with context / spatial / Identity,
  TopK=2 and sparse input masking.
- Context retains two-level db1 DWT and all-subband IWT reconstruction.
- Existing `experts_multi_low` best checkpoint loads strictly after removing
  420 unused context `llconv` / `hconv` tensors.
- With those same active weights, all four predictions on a fixed
  `[2,3,64,64]` input match the pre-cleanup context-only model **bit for bit**
  (maximum absolute error 0 for all outputs).
- Context output for a fixed odd `[2,C,15,17]` input is also bitwise identical.
- Gradient finiteness, sample independence, sparse input masking, TopK bounds,
  and checkpoint rejection for four-expert routers are covered by the tests.
- CUDA was not initialized; no training experiment was started.

Reference checkpoint SHA256:
`28661c0077a5646367374a641eb27309ddfb47f359bfd180594c8c1746f99370c`.

The compatibility conversion preserves inference with the existing weights;
it does not preserve optimizer state or claim identical from-scratch training.
Removing unused modules changes the initialization RNG consumption order.
