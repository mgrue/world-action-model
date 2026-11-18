"""Pose readout (ticket 01) on a saved prototype target encoder: uv run python readout_ckpt.py NAME"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import jepa_proto as J
from toycar.dataset import open_dataset
from wam import compressor as C

name = sys.argv[1]
d = J.OUT / f"jepa_{name}"
ck = torch.load(d / "target.pt", map_location="cpu")
a = ck["args"]
dev = C.pick_device(None)
enc = J.Encoder(a["width"], a["depth"], a["heads"], a["out_dim"]).to(dev)
enc.load_state_dict(ck["target"]); enc.eval()
ds = open_dataset("data/step3-v1", check=False)
res = J.pose_readout(enc, ds.split("train"), ds.split("val"), dev)
(d / "readout.json").write_text(json.dumps(res, indent=1))
