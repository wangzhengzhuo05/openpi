import pandas as pd
from pathlib import Path
import sys

# 数据集路径
dataset_path = Path("/root/autodl-tmp/huggingface/lerobot/Coil1987121/calvin_lerobot_task_ABCD_D_training_part1")
data_dir = dataset_path / "data"

print(f"Dataset path: {dataset_path}")

# 要删除的 episodes
episodes_to_remove = [11858, 11859, 11860]
print(f"\n=== Removing episodes: {episodes_to_remove} ===")

# 读取所有数据
print("\nReading all parquet files...")
parquet_files = list(data_dir.rglob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

all_data = []
for i, parquet_file in enumerate(sorted(parquet_files)):
    if i % 50 == 0:
        print(f"  Reading file {i+1}/{len(parquet_files)}")
    df = pd.read_parquet(parquet_file)
    all_data.append(df)

print(f"\nConcatenating {len(all_data)} dataframes...")
full_df = pd.concat(all_data, ignore_index=True)

print(f"Original data:")
print(f"  Total rows: {len(full_df)}")
print(f"  Total episodes: {full_df['episode_index'].nunique()}")
print(f"  Episode range: {full_df['episode_index'].min()} to {full_df['episode_index'].max()}")

# 统计要删除的数据
removed_rows = full_df[full_df['episode_index'].isin(episodes_to_remove)]
print(f"\nData to be removed:")
print(f"  Episodes: {episodes_to_remove}")
print(f"  Rows: {len(removed_rows)} ({len(removed_rows) / len(full_df) * 100:.2f}%)")

# 删除指定 episodes
clean_df = full_df[~full_df['episode_index'].isin(episodes_to_remove)].copy()

print(f"\nAfter removal:")
print(f"  Total rows: {len(clean_df)}")
print(f"  Total episodes: {clean_df['episode_index'].nunique()}")
print(f"  Removed: {len(full_df) - len(clean_df)} rows")

# 重新编号 episode_index（从 0 开始连续）
print(f"\nReindexing episodes...")
unique_episodes = sorted(clean_df['episode_index'].unique())
episode_mapping = {old: new for new, old in enumerate(unique_episodes)}
clean_df['episode_index'] = clean_df['episode_index'].map(episode_mapping)

print(f"  New episode range: {clean_df['episode_index'].min()} to {clean_df['episode_index'].max()}")
print(f"  New total episodes: {clean_df['episode_index'].nunique()}")

# 备份原数据
backup_dir = dataset_path / "data_backup"
if not backup_dir.exists():
    print(f"\nBacking up original data to: {backup_dir}")
    import shutil
    shutil.copytree(data_dir, backup_dir)
    print(f"  ✅ Backup complete")
else:
    print(f"\nBackup already exists at: {backup_dir}")

# 清空原 data 目录
print(f"\nClearing original data directory...")
import shutil
for item in data_dir.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

# 保存清理后的数据（保持原有结构）
print(f"\nSaving cleaned data...")

# 创建 chunk 子目录结构
num_chunks = 182  # 保持原有的分片数
rows_per_chunk = len(clean_df) // num_chunks + 1

chunk_idx = 0
for chunk_num in range(num_chunks):
    start_idx = chunk_num * rows_per_chunk
    end_idx = min(start_idx + rows_per_chunk, len(clean_df))
    
    if start_idx >= len(clean_df):
        break
    
    chunk_data = clean_df.iloc[start_idx:end_idx]
    
    # 创建 chunk 子目录
    chunk_dir = data_dir / f"chunk-{chunk_num:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存到子目录
    output_file = chunk_dir / "data.parquet"
    chunk_data.to_parquet(output_file, index=False)
    
    if chunk_num % 20 == 0:
        print(f"  Saved chunk {chunk_num}/{num_chunks}")
    
    chunk_idx += 1

print(f"\n✅ Cleaning complete!")
print(f"\nSummary:")
print(f"  Original episodes: 13696")
print(f"  Removed episodes: {len(episodes_to_remove)}")
print(f"  Remaining episodes: {clean_df['episode_index'].nunique()}")
print(f"  Original rows: {len(full_df)}")
print(f"  Remaining rows: {len(clean_df)}")
print(f"  Backup location: {backup_dir}")
print(f"\nYou can now run: uv run scripts/compute_norm_stats.py --config-name pi0_calvin_scratch")