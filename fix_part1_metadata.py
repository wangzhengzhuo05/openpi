import json
from pathlib import Path
import pandas as pd

dataset_path = Path("/root/autodl-tmp/huggingface/lerobot/Coil1987121/calvin_lerobot_task_ABCD_D_training_part1")

print("=== Scanning actual files ===")
actual_files = sorted(list(dataset_path.glob("data/**/*.parquet")))
print(f"Found {len(actual_files)} parquet files")

# 读取所有数据来统计实际情况
print("\nReading files to get statistics...")
all_episodes = set()
total_frames = 0

for i, pf in enumerate(actual_files):
    if i % 1000 == 0:
        print(f"  Processing {i+1}/{len(actual_files)}")
    try:
        df = pd.read_parquet(pf)
        if 'episode_index' in df.columns:
            all_episodes.update(df['episode_index'].unique())
        total_frames += len(df)
    except Exception as e:
        print(f"  Error reading {pf.name}: {e}")

actual_num_episodes = len(all_episodes)
max_episode = max(all_episodes) if all_episodes else 0
min_episode = min(all_episodes) if all_episodes else 0

print(f"\n=== Actual dataset statistics ===")
print(f"  Total episodes: {actual_num_episodes}")
print(f"  Episode range: {min_episode} to {max_episode}")
print(f"  Total frames: {total_frames}")
print(f"  Episode indices: {sorted(list(all_episodes))[:10]}...{sorted(list(all_episodes))[-10:]}")

# 读取并备份原始 info.json
info_path = dataset_path / "meta" / "info.json"
info_backup = dataset_path / "meta" / "info.json.backup"

with open(info_path, 'r') as f:
    info = json.load(f)

print(f"\n=== Original info.json ===")
print(f"  total_episodes: {info['total_episodes']}")
print(f"  total_frames: {info['total_frames']}")

# 备份
import shutil
shutil.copy(info_path, info_backup)
print(f"\n✅ Backed up to: {info_backup}")

# 更新 info.json
info['total_episodes'] = actual_num_episodes
info['total_frames'] = total_frames

with open(info_path, 'w') as f:
    json.dump(info, f, indent=4)

print(f"\n=== Updated info.json ===")
print(f"  total_episodes: {info['total_episodes']}")
print(f"  total_frames: {info['total_frames']}")

# 更新 episodes.jsonl（如果存在）
episodes_path = dataset_path / "meta" / "episodes.jsonl"
if episodes_path.exists():
    print(f"\n=== Updating episodes.jsonl ===")
    episodes_backup = dataset_path / "meta" / "episodes.jsonl.backup"
    shutil.copy(episodes_path, episodes_backup)
    
    # 读取并过滤
    valid_episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            ep = json.loads(line)
            if ep['episode_index'] in all_episodes:
                valid_episodes.append(ep)
    
    print(f"  Original episodes: {len(valid_episodes) + (len(all_episodes) if len(valid_episodes) == 0 else 0)}")
    print(f"  Valid episodes: {len(valid_episodes)}")
    
    # 如果 episodes.jsonl 是空的或不匹配，重新生成
    if len(valid_episodes) != actual_num_episodes:
        print(f"  ⚠️ episodes.jsonl doesn't match actual data, will need regeneration")
        print(f"  For now, creating minimal version...")
        
        valid_episodes = [
            {
                "episode_index": int(ep_idx),
                "tasks": []
            }
            for ep_idx in sorted(all_episodes)
        ]
    
    with open(episodes_path, 'w') as f:
        for ep in valid_episodes:
            f.write(json.dumps(ep) + '\n')
    
    print(f"  ✅ Updated episodes.jsonl")

print("\n✅ Metadata fixed!")
print("\nYou can now run:")
print("  uv run scripts/compute_norm_stats.py --config-name pi0_calvin_scratch_repeat")
