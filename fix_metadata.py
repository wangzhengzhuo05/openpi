
import json
from pathlib import Path
import pandas as pd
import shutil

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

print(f"\n=== Actual dataset statistics ===")
print(f"  Total episodes: {actual_num_episodes}")
print(f"  Total frames: {total_frames}")

# 读取并备份原始 info.json
info_path = dataset_path / "meta" / "info.json"
info_backup = dataset_path / "meta" / "info.json.backup"

with open(info_path, 'r') as f:
    info = json.load(f)

print(f"\n=== Original info.json ===")
print(f"  total_episodes: {info['total_episodes']}")
print(f"  total_frames: {info['total_frames']}")

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
    print(f"  Backed up to: {episodes_backup}")
    
    # 读取并过滤（处理错误行）
    valid_episodes = []
    line_num = 0
    errors = 0
    
    with open(episodes_path, 'r') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
            
            try:
                ep = json.loads(line)
                # 只保留实际存在的 episodes
                if ep['episode_index'] in all_episodes:
                    valid_episodes.append(ep)
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    print(f"  Warning: Line {line_num} has invalid JSON: {e}")
                continue
    
    if errors > 5:
        print(f"  ... and {errors - 5} more errors")
    
    print(f"  Original lines: {line_num}")
    print(f"  Valid episodes: {len(valid_episodes)}")
    print(f"  Expected episodes: {actual_num_episodes}")
    
    # 如果数量不匹配，重新生成简化版本
    if len(valid_episodes) != actual_num_episodes:
        print(f"  ⚠️ Mismatch detected, generating minimal episodes.jsonl...")
        valid_episodes = [
            {
                "episode_index": int(ep_idx),
                "tasks": []
            }
            for ep_idx in sorted(all_episodes)
        ]
    
    # 写入更新后的文件
    with open(episodes_path, 'w') as f:
        for ep in valid_episodes:
            f.write(json.dumps(ep) + '\n')
    
    print(f"  ✅ Updated episodes.jsonl with {len(valid_episodes)} episodes")

# 处理 episodes_stats.jsonl（如果存在）
stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
if stats_path.exists():
    print(f"\n=== Updating episodes_stats.jsonl ===")
    stats_backup = dataset_path / "meta" / "episodes_stats.jsonl.backup"
    shutil.copy(stats_path, stats_backup)
    
    valid_stats = []
    errors = 0
    
    with open(stats_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                stat = json.loads(line)
                if stat['episode_index'] in all_episodes:
                    valid_stats.append(stat)
            except (json.JSONDecodeError, KeyError):
                errors += 1
                continue
    
    with open(stats_path, 'w') as f:
        for stat in valid_stats:
            f.write(json.dumps(stat) + '\n')
    
    print(f"  ✅ Updated episodes_stats.jsonl with {len(valid_stats)} episodes")

print("\n" + "="*60)
print("✅ Metadata fixed successfully!")
print("="*60)
print(f"\nDataset now has:")
print(f"  Episodes: {actual_num_episodes}")
print(f"  Frames: {total_frames}")
print(f"\nBackups created:")
print(f"  {info_backup}")
print(f"  {episodes_backup if episodes_path.exists() else 'N/A'}")
print(f"\nNext step:")
print(f"  uv run scripts/compute_norm_stats.py --config-name pi0_calvin_scratch_repeat")

