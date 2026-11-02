import json
from pathlib import Path
old=Path("/workspace/local-vlm/plancarry_a1_technical_qwen_kaggle_v5_20261101/original_qwen_kv_3reset_streamed_dev_smoke.ipynb")
n=json.loads(old.read_text())
new={"cells":n["cells"][:2],"metadata":n.get("metadata",{}),"nbformat":n["nbformat"],"nbformat_minor":n.get("nbformat_minor",5)}
src=Path("v9_dev_ninearm_cell_v2.py").read_text()
new["cells"].append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.splitlines(keepends=True)})
Path("plancarry_qwen_9arm_fourcut_dev_v1.ipynb").write_text(json.dumps(new,ensure_ascii=False))
meta=json.loads(Path("/workspace/local-vlm/plancarry_a1_technical_qwen_kaggle_v5_20261101/kernel-metadata.json").read_text())
meta.update({"id":"chuckies/plancarry-qwen-9-arm-four-cut-dev","title":"PlanCarry Qwen 9 Arm Four Cut DEV","code_file":"plancarry_qwen_9arm_fourcut_dev_v1.ipynb"})
Path("kernel-metadata.json").write_text(json.dumps(meta,indent=2))
assert len(new["cells"])==3
assert "PREFLIGHT_FAILED" in src
print("DEV_NOTEBOOK_BUILT",Path("plancarry_qwen_9arm_fourcut_dev_v1.ipynb").stat().st_size)
