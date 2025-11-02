"""
Enhanced diagnostic script to check CALVIN NPZ file structure
Focuses on the first episode that HAS language annotation
"""
import numpy as np
import os

data_dir = "/root/autodl-tmp/dataset/calvin_debug_dataset/training"

# Load language annotations
lang_path = os.path.join(data_dir, "lang_annotations/auto_lang_ann.npy")
lang_data = np.load(lang_path, allow_pickle=True).item()
lang_tasks = lang_data["language"]["task"]
lang_ann = lang_data["language"]["ann"]
lang_ranges = lang_data["info"]["indx"]

# Build episode -> language mapping
episode_to_lang = {}
for (start, end), task_name, sentence in zip(lang_ranges, lang_tasks, lang_ann):
    for ep_id in range(start, end):
        episode_to_lang[ep_id] = {"task": task_name, "instruction": sentence}

print(f"Total episodes with language annotation: {len(episode_to_lang)}")
print(f"Episode indices with annotations: {sorted(list(episode_to_lang.keys()))[:10]}...")

# Find first NPZ file with annotation
npz_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
print(f"\nTotal NPZ files: {len(npz_files)}")

first_annotated = None
for npz_file in npz_files:
    episode_idx = int(npz_file.split("_")[1].split(".")[0])
    if episode_idx in episode_to_lang:
        first_annotated = npz_file
        first_annotated_idx = episode_idx
        break

if first_annotated is None:
    print("ERROR: No annotated episodes found!")
    exit(1)

print(f"\n{'='*70}")
print(f"ANALYZING FIRST ANNOTATED EPISODE: {first_annotated}")
print(f"Episode index: {first_annotated_idx}")
print(f"Task: {episode_to_lang[first_annotated_idx]['task']}")
print(f"Instruction: {episode_to_lang[first_annotated_idx]['instruction']}")
print(f"{'='*70}\n")

# Load the file
file_path = os.path.join(data_dir, first_annotated)
data = dict(np.load(file_path))

print("ARRAY SHAPES AND TYPES:")
print("-" * 70)
for key in sorted(data.keys()):
    arr = data[key]
    print(f"{key:20s}: shape={str(arr.shape):20s} dtype={arr.dtype}")

print("\n" + "="*70)
print("DETAILED ANALYSIS")
print("="*70)

# Check if there's a consistent time dimension
shapes = {key: data[key].shape for key in data.keys()}

# Check actions specifically
if "actions" in data:
    print(f"\nACTIONS array:")
    print(f"  Shape: {data['actions'].shape}")
    print(f"  Dtype: {data['actions'].dtype}")
    if len(data['actions'].shape) == 1:
        print(f"  This is 1D - single timestep! Length: {len(data['actions'])}")
        print(f"  First few values: {data['actions'][:5]}")
    elif len(data['actions'].shape) == 2:
        print(f"  This is 2D - multiple timesteps")
        print(f"  Likely format: (timesteps, action_dim) = {data['actions'].shape}")
        print(f"  OR could be: (action_dim, timesteps) = {data['actions'].shape}")
        print(f"  First row: {data['actions'][0]}")
        print(f"  First column: {data['actions'][:, 0]}")

# Check rgb_static
if "rgb_static" in data:
    print(f"\nRGB_STATIC array:")
    print(f"  Shape: {data['rgb_static'].shape}")
    print(f"  Dtype: {data['rgb_static'].dtype}")
    if len(data['rgb_static'].shape) == 3:
        print(f"  This is 3D - likely (H, W, C) = {data['rgb_static'].shape}")
        print(f"  No time dimension!")
    elif len(data['rgb_static'].shape) == 4:
        print(f"  This is 4D - has time dimension")
        print(f"  Could be (T, H, W, C) or (H, W, C, T) or other permutations")

# Check rgb_gripper
if "rgb_gripper" in data:
    print(f"\nRGB_GRIPPER array:")
    print(f"  Shape: {data['rgb_gripper'].shape}")
    print(f"  Dtype: {data['rgb_gripper'].dtype}")

# Check rgb_tactile
if "rgb_tactile" in data:
    print(f"\nRGB_TACTILE array:")
    print(f"  Shape: {data['rgb_tactile'].shape}")
    print(f"  Dtype: {data['rgb_tactile'].dtype}")
    if len(data['rgb_tactile'].shape) == 3:
        print(f"  This is 3D - likely (H, W, C) where C=6 for left+right")
        print(f"  No time dimension!")

# Check depth arrays
if "depth_static" in data:
    print(f"\nDEPTH_STATIC array:")
    print(f"  Shape: {data['depth_static'].shape}")
    print(f"  Dtype: {data['depth_static'].dtype}")

if "depth_tactile" in data:
    print(f"\nDEPTH_TACTILE array:")
    print(f"  Shape: {data['depth_tactile'].shape}")
    print(f"  Dtype: {data['depth_tactile'].dtype}")

# Try to infer if this is a single-timestep episode
print("\n" + "="*70)
print("INFERENCE:")
print("="*70)

# Count dimensions
dim_counts = {}
for key, shape in shapes.items():
    ndim = len(shape)
    if ndim not in dim_counts:
        dim_counts[ndim] = []
    dim_counts[ndim].append(key)

for ndim in sorted(dim_counts.keys()):
    print(f"\n{ndim}D arrays: {dim_counts[ndim]}")

# Check if all "temporal" arrays have the same first dimension
temporal_keys = ["actions", "rel_actions", "robot_obs", "scene_obs"]
if all(k in data for k in temporal_keys):
    first_dims = [data[k].shape[0] if len(data[k].shape) > 0 else 0 for k in temporal_keys]
    print(f"\nFirst dimension of temporal arrays: {dict(zip(temporal_keys, first_dims))}")
    if len(set(first_dims)) == 1:
        print(f"  ✓ All have same first dimension: {first_dims[0]} (likely the number of timesteps)")
    else:
        print(f"  ⚠ Different first dimensions - need to investigate!")