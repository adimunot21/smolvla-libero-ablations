"""Phase 1 / Step E — Inspect HuggingFaceVLA/libero dataset.

Critical check: feature names in the dataset must match what the LIBERO
sim emits (we verified those in scripts/01_inspect_libero_raw.py).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from dotenv import load_dotenv
load_dotenv()  # loads HF_TOKEN from .env

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "HuggingFaceVLA/libero"
print(f"Loading {REPO} (first time will download metadata + 1 episode)...")

# Load only episode 0 to keep download tiny
ds = LeRobotDataset(REPO, episodes=[0])
print(f"\n=== Dataset metadata ===")
print(f"  num_episodes:   {ds.meta.total_episodes}")
print(f"  num_frames:     {ds.meta.total_frames}")
print(f"  fps:            {ds.meta.fps}")
print(f"  robot_type:     {ds.meta.robot_type}")
print(f"  tasks:          {len(ds.meta.tasks)}")

print(f"\n=== Features ===")
for k, v in ds.meta.features.items():
    print(f"  {k:40s}  shape={v.get('shape', '-')}  dtype={v.get('dtype', '-')}")

print(f"\n=== Sample frame (episode 0, frame 0) ===")
sample = ds[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(f"  {k:40s}  shape={tuple(v.shape)}  dtype={v.dtype}")
    else:
        print(f"  {k:40s}  type={type(v).__name__}  value={str(v)[:80]}")

# Most important check: image keys
print(f"\n=== Image keys present ===")
img_keys = [k for k in ds.meta.features if "image" in k.lower() or "observation.images" in k]
for k in img_keys:
    print(f"  {k}")

print("\nSTEP E PASSED" if img_keys else "WARNING: no image keys found")
