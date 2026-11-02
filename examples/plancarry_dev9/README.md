# PlanCarry nine-arm technical DEV

This directory records the isolated historical DEV integration of Qwen2.5-7B-Instruct on Kaggle T4×2. Version 1 ran four cuts (0/1/2/3), nine named branches per cut, and 36 model forwards. Version 2 adds raw model replies and an append-only continuous diagnostic. It is a technical smoke test, not Stage1 U scientific execution.

- Kaggle notebook: https://www.kaggle.com/code/chuckies/plancarry-qwen-9-arm-four-cut-dev
- Output SHA-256: `04d1f746025f1fc136ec868423ce8364e6bcaf8f92dde404745a89d3b1e328e3`
- Independent receipt check SHA-256: `299b08d6c6e4ccd8bb8331f00e6381af1761a1490174214e8db367238c0dd873`
- Version 2 output SHA-256 (36 raw replies): `ff80f20b2d1454e9fe247ce198a012f5da53fb26128f22da97beecd2ed13725e`
- Version 2 independent validation SHA-256: `a492f926bc2066c715267dd6931645848f83c969daab3c7a9ed941eb9287a6f2`
- Protected Research OS source bundle: `sha256:d63e52f9605243125434f74d23d53e8969e4ba8404d0fef727ea77a23c869cbf`

`streamed_prefill_v9.py` is the bounded prefill implementation. `plancarry_qwen25_stage1u_v9_streamed_future_NO_SCIENCE.py` and `plancarry_stage1u_v9_bound_future_runner_NO_SCIENCE.py` are guarded future integration source; they require the frozen v5/v8 source and native receipts retained in Research OS. The Stage1 gate remains `PREFLIGHT_FAILED`.

The Kaggle DEV notebook uses the historical exposed native public trace and never changes the Stage1 gate. Its uninterrupted branch is not a matched reset comparator. It does not perform native model-driven action rollouts or establish a scientific effect. The `v9_dev_ninearm_cell_v2.py` source is the current DEV cell.
