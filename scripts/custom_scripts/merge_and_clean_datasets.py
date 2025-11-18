import pandas as pd
from pathlib import Path
import shutil
import json
import numpy as np
from datetime import datetime
import gc  # 垃圾回收

def check_episode_timestamps(episode_data):
    """检查单个 episode 的时间戳问题"""
    episode_data = episode_data.sort_values('frame_index').reset_index(drop=True)
    timestamps = episode_data['timestamp'].values
    
    if len(timestamps) <= 1:
        return True, None
    
    diffs = timestamps[1:] - timestamps[:-1]
    issues = []
    
    if (diffs < 0).any():
        issues.append(f"negative_diffs={(diffs < 0).sum()}")
    if (diffs == 0).any():
        issues.append(f"zero_diffs={(diffs == 0).sum()}")
    if timestamps[0] != 0.0:
        issues.append(f"start={timestamps[0]:.4f}")
    
    if issues:
        return False, "; ".join(issues)
    return True, None


def load_dataset_batch(data_dir, batch_size=1000):
    """分批加载数据集，避免内存溢出"""
    parquet_files = list(data_dir.rglob("*.parquet"))
    total_files = len(parquet_files)
    
    print(f"Found {total_files} parquet files")
    print(f"Loading in batches of {batch_size} files...")
    
    all_chunks = []
    for i in range(0, total_files, batch_size):
        batch_files = sorted(parquet_files)[i:i+batch_size]
        print(f"  Loading batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1} "
              f"(files {i+1}-{min(i+batch_size, total_files)})")
        
        batch_data = []
        for pf in batch_files:
            try:
                df = pd.read_parquet(pf)
                batch_data.append(df)
            except Exception as e:
                print(f"    Error reading {pf.name}: {e}")
        
        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            all_chunks.append(batch_df)
            print(f"    Batch loaded: {len(batch_df):,} rows")
            
            # 清理内存
            del batch_data
            gc.collect()
    
    print(f"\nConcatenating {len(all_chunks)} batches...")
    full_df = pd.concat(all_chunks, ignore_index=True)
    
    # 最终清理
    del all_chunks
    gc.collect()
    
    return full_df


def load_dataset(dataset_path, dataset_name):
    """加载数据集（分批处理）"""
    data_dir = dataset_path / "data"
    
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name}...")
    print(f"Path: {dataset_path}")
    print(f"{'='*60}")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return None
    
    # 使用分批加载
    full_df = load_dataset_batch(data_dir, batch_size=500)  # 每次处理500个文件
    
    print(f"\nDataset info:")
    print(f"  Total rows: {len(full_df):,}")
    print(f"  Total episodes: {full_df['episode_index'].nunique():,}")
    print(f"  Episode range: {full_df['episode_index'].min()} to {full_df['episode_index'].max()}")
    
    return full_df


def check_all_episodes(df, dataset_name):
    """检查所有 episodes 的时间戳"""
    print(f"\n{'='*60}")
    print(f"Checking timestamps for {dataset_name}...")
    print(f"{'='*60}")
    
    problem_episodes = []
    unique_episodes = sorted(df['episode_index'].unique())
    total_episodes = len(unique_episodes)
    
    for i, episode_idx in enumerate(unique_episodes):
        if i % 1000 == 0:
            print(f"  Checking episode {i+1}/{total_episodes}")
        
        episode_data = df[df['episode_index'] == episode_idx]
        is_ok, issue_desc = check_episode_timestamps(episode_data)
        
        if not is_ok:
            problem_episodes.append({
                'episode_index': episode_idx,
                'num_frames': len(episode_data),
                'issues': issue_desc
            })
    
    print(f"\nResults:")
    print(f"  Total episodes checked: {total_episodes:,}")
    print(f"  Problem episodes found: {len(problem_episodes):,} ({len(problem_episodes)/total_episodes*100:.2f}%)")
    
    if len(problem_episodes) > 0:
        print(f"\nFirst 20 problem episodes:")
        for ep in problem_episodes[:20]:
            print(f"  Episode {ep['episode_index']}: {ep['num_frames']} frames, issues: {ep['issues']}")
        if len(problem_episodes) > 20:
            print(f"  ... and {len(problem_episodes) - 20} more")
    
    return problem_episodes


def merge_and_save_streaming(df1, df2, problem_episodes_1, problem_episodes_2, output_path, source_path):
    """流式合并和保存，避免内存问题"""
    print(f"\n{'='*60}")
    print(f"Merging and saving (streaming mode)...")
    print(f"{'='*60}")
    
    # 删除问题 episodes
    problem_indices_1 = set([ep['episode_index'] for ep in problem_episodes_1])
    problem_indices_2 = set([ep['episode_index'] for ep in problem_episodes_2])
    
    print(f"\nFiltering Part 1...")
    clean_df1 = df1[~df1['episode_index'].isin(problem_indices_1)].copy()
    print(f"  Remaining: {len(clean_df1):,} rows, {clean_df1['episode_index'].nunique():,} episodes")
    
    print(f"\nFiltering Part 2...")
    clean_df2 = df2[~df2['episode_index'].isin(problem_indices_2)].copy()
    print(f"  Remaining: {len(clean_df2):,} rows, {clean_df2['episode_index'].nunique():,} episodes")
    
    # 重新编号
    print(f"\nReindexing episodes...")
    unique_episodes_1 = sorted(clean_df1['episode_index'].unique())
    episode_mapping_1 = {old: new for new, old in enumerate(unique_episodes_1)}
    clean_df1['episode_index'] = clean_df1['episode_index'].map(episode_mapping_1)
    
    offset = len(unique_episodes_1)
    unique_episodes_2 = sorted(clean_df2['episode_index'].unique())
    episode_mapping_2 = {old: new + offset for new, old in enumerate(unique_episodes_2)}
    clean_df2['episode_index'] = clean_df2['episode_index'].map(episode_mapping_2)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = output_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 保存 Part 1
    print(f"\nSaving Part 1 data...")
    chunk_size = 5000
    chunk_num = 0
    for i in range(0, len(clean_df1), chunk_size):
        chunk = clean_df1.iloc[i:i+chunk_size]
        chunk_dir = data_dir / f"chunk-{chunk_num:03d}"
        chunk_dir.mkdir(exist_ok=True)
        chunk.to_parquet(chunk_dir / "data.parquet", index=False)
        if chunk_num % 10 == 0:
            print(f"  Saved chunk {chunk_num}")
        chunk_num += 1
    
    # 清理 Part 1
    del clean_df1
    gc.collect()
    
    # 保存 Part 2
    print(f"\nSaving Part 2 data...")
    for i in range(0, len(clean_df2), chunk_size):
        chunk = clean_df2.iloc[i:i+chunk_size]
        chunk_dir = data_dir / f"chunk-{chunk_num:03d}"
        chunk_dir.mkdir(exist_ok=True)
        chunk.to_parquet(chunk_dir / "data.parquet", index=False)
        if chunk_num % 10 == 0:
            print(f"  Saved chunk {chunk_num}")
        chunk_num += 1
    
    # 复制 meta
    print(f"\nCopying metadata...")
    source_meta = source_path / "meta"
    if source_meta.exists():
        shutil.copytree(source_meta, output_path / "meta", dirs_exist_ok=True)
    
    print(f"\n✅ Saved {chunk_num} chunks to: {output_path}")
    
    total_episodes = len(unique_episodes_1) + len(unique_episodes_2)
    total_rows = len(clean_df2)  # 还在内存中
    
    return total_episodes, total_rows


def main():
    part1_path = Path("/root/autodl-tmp/huggingface/lerobot/Coil1987121/calvin_lerobot_task_ABCD_D_training_part1")
    part2_path = Path("/root/autodl-tmp/huggingface/lerobot/Coil1987121/calvin_lerobot_task_ABCD_D_training_part2")
    output_path = Path("/root/autodl-tmp/huggingface/lerobot/Coil1987121/calvin_lerobot_task_ABCD_D_training_merged")
    
    print(f"{'='*60}")
    print(f"CALVIN Dataset Merger and Cleaner (Memory Optimized)")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载
    df1 = load_dataset(part1_path, "Part 1")
    if df1 is None:
        return
    
    df2 = load_dataset(part2_path, "Part 2")
    if df2 is None:
        return
    
    # 2. 检查
    problem_episodes_1 = check_all_episodes(df1, "Part 1")
    problem_episodes_2 = check_all_episodes(df2, "Part 2")
    
    # 3. 合并和保存
    total_episodes, total_rows = merge_and_save_streaming(
        df1, df2, problem_episodes_1, problem_episodes_2, output_path, part1_path
    )
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total episodes: {total_episodes:,}")
    print(f"Removed: {len(problem_episodes_1) + len(problem_episodes_2):,} episodes")
    print(f"Output: {output_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ All done!")


if __name__ == "__main__":
    main()