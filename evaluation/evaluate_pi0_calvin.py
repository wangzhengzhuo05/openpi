"""
Pi0 CALVIN 模型评估脚本 - 最终版 v2
支持语言指令加载

uv run /path/to/evaluate_pi0_calvin.py \
    --checkpoint_dir /root/autodl-tmp/openpi/checkpoints/pi0_calvin_scratch/calvin_full/21000 \
    --dataset_path /root/autodl-tmp/huggingface/lerobot/Coil1987121/calvin_lerobot_task_ABCD_D_validation \
    --config_name pi0_calvin_scratch \
    --num_samples 100 \
    --action_horizon 10
"""

import sys
import json
import os
import io
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# OpenPI 路径配置
OPENPI_ROOT = os.environ.get('OPENPI_ROOT', '/root/autodl-tmp/openpi')
sys.path.insert(0, f'{OPENPI_ROOT}/src')
sys.path.insert(0, OPENPI_ROOT)


# ============================================================================
# 图像解析
# ============================================================================

def parse_image(image: Any, dataset_path: Path = None) -> np.ndarray:
    """解析图像为 (H, W, C) uint8 格式"""
    from PIL import Image
    
    if image is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # LeRobot 字典格式 {'bytes': b'...', 'path': '...'}
    if isinstance(image, dict):
        if 'bytes' in image and image['bytes'] is not None:
            try:
                pil_img = Image.open(io.BytesIO(image['bytes']))
                return np.array(pil_img.convert('RGB'), dtype=np.uint8)
            except:
                pass
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # bytes 数据
    if isinstance(image, (bytes, bytearray)):
        try:
            pil_img = Image.open(io.BytesIO(image))
            return np.array(pil_img.convert('RGB'), dtype=np.uint8)
        except:
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # numpy array
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).clip(0, 255).astype(np.uint8)
    return image.astype(np.uint8)


def parse_array(data: Any, expected_dim: int = 7) -> np.ndarray:
    """解析数组"""
    if data is None:
        return np.zeros(expected_dim, dtype=np.float32)
    if isinstance(data, (list, tuple)):
        return np.array(data, dtype=np.float32).flatten()
    if isinstance(data, np.ndarray):
        return data.astype(np.float32).flatten()
    return np.zeros(expected_dim, dtype=np.float32)


# ============================================================================
# 语言指令加载
# ============================================================================

def load_task_instructions(dataset_path: Path) -> Dict[int, str]:
    """
    从 meta/tasks.jsonl 加载任务描述
    
    Returns:
        task_map: {task_index: task_description}
    """
    task_map = {}
    
    # 可能的任务文件路径
    possible_paths = [
        dataset_path / "meta" / "tasks.jsonl",
        dataset_path / "tasks.jsonl",
        dataset_path / "meta" / "tasks.json",
    ]
    
    task_file = None
    for p in possible_paths:
        if p.exists():
            task_file = p
            break
    
    if task_file is None:
        print("⚠ 未找到任务描述文件 (tasks.jsonl)")
        return task_map
    
    print(f"加载任务描述: {task_file}")
    
    with open(task_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                task_idx = item.get('task_index')
                task_desc = item.get('task', '')
                if task_idx is not None:
                    task_map[int(task_idx)] = task_desc
            except json.JSONDecodeError:
                continue
    
    print(f"  加载了 {len(task_map)} 个任务描述")
    
    # 打印几个示例
    if task_map:
        examples = list(task_map.items())[:3]
        for idx, desc in examples:
            print(f"  - [{idx}] {desc}")
    
    return task_map


# ============================================================================
# 评估指标
# ============================================================================

def compute_rotation_error_np(q_pred, q_gt):
    """计算四元数角度误差 (度)"""
    q_pred = q_pred / (np.linalg.norm(q_pred, axis=-1, keepdims=True) + 1e-8)
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=-1, keepdims=True) + 1e-8)
    dot = np.abs(np.sum(q_pred * q_gt, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.rad2deg(2 * np.arccos(dot))


def evaluate_trajectory_np(pred_traj, gt_traj):
    """评估轨迹质量"""
    metrics = {}
    
    # 基础误差
    metrics['traj_mse'] = float(np.mean((pred_traj - gt_traj) ** 2))
    metrics['traj_mae'] = float(np.mean(np.abs(pred_traj - gt_traj)))
    metrics['traj_rmse'] = float(np.sqrt(metrics['traj_mse']))
    
    # 位置误差
    pos_pred, pos_gt = pred_traj[:, :, :3], gt_traj[:, :, :3]
    metrics['pos_mse'] = float(np.mean((pos_pred - pos_gt) ** 2))
    metrics['pos_rmse_cm'] = float(np.sqrt(metrics['pos_mse']) * 100)
    
    # FDE / ADE
    fde = np.linalg.norm(pos_pred[:, -1] - pos_gt[:, -1], axis=-1)
    metrics['fde_mean_cm'] = float(np.mean(fde) * 100)
    metrics['fde_median_cm'] = float(np.median(fde) * 100)
    metrics['fde_std_cm'] = float(np.std(fde) * 100)
    
    ade = np.mean(np.linalg.norm(pos_pred - pos_gt, axis=-1), axis=1)
    metrics['ade_mean_cm'] = float(np.mean(ade) * 100)
    metrics['ade_median_cm'] = float(np.median(ade) * 100)
    
    # 旋转误差
    if pred_traj.shape[-1] >= 7:
        rot_err = compute_rotation_error_np(pred_traj[:, -1, 3:7], gt_traj[:, -1, 3:7])
        metrics['rot_error_mean_deg'] = float(np.mean(rot_err))
        metrics['rot_error_median_deg'] = float(np.median(rot_err))
        metrics['rot_error_std_deg'] = float(np.std(rot_err))
        metrics['rot_mse'] = float(np.mean((pred_traj[:, :, 3:7] - gt_traj[:, :, 3:7]) ** 2))
        
        # 成功率
        metrics['sr_1cm_2deg'] = float(np.mean((fde < 0.01) & (rot_err < 2.0)))
        metrics['sr_2cm_5deg'] = float(np.mean((fde < 0.02) & (rot_err < 5.0)))
        metrics['sr_3cm_10deg'] = float(np.mean((fde < 0.03) & (rot_err < 10.0)))
        metrics['sr_5cm_15deg'] = float(np.mean((fde < 0.05) & (rot_err < 15.0)))
    
    # 每维度 MSE
    per_dim_mse = np.mean((pred_traj - gt_traj) ** 2, axis=(0, 1))
    dim_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper'][:pred_traj.shape[-1]]
    for i, name in enumerate(dim_names):
        metrics[f'mse_{name}'] = float(per_dim_mse[i])
    
    return metrics


# ============================================================================
# 数据集类
# ============================================================================

def find_column(df, patterns):
    for pattern in patterns:
        if pattern in df.columns:
            return pattern
        for col in df.columns:
            if pattern.lower() in col.lower():
                return col
    return None


class CALVINEvalDataset:
    """CALVIN 评估数据集"""
    
    def __init__(
        self, 
        dataset_path: str, 
        max_episodes: int = None, 
        verbose: bool = True
    ):
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        
        # 加载任务描述
        self.task_map = load_task_instructions(self.dataset_path)
        
        # 加载数据
        self.data = self._load_data()
        self._detect_columns()
        self.episodes = self._build_episodes(max_episodes)
        
        if verbose:
            print(f"✓ 数据集: {len(self.episodes)} episodes, {len(self.data)} frames")
    
    def _load_data(self):
        for data_dir in [self.dataset_path / "data", self.dataset_path]:
            if not data_dir.exists():
                continue
            parquet_files = list(data_dir.rglob("*.parquet"))
            if parquet_files:
                if self.verbose:
                    print(f"从 {data_dir} 加载 {len(parquet_files)} 个 parquet 文件...")
                dfs = [pd.read_parquet(f) for f in tqdm(parquet_files, desc="读取数据", disable=not self.verbose)]
                return pd.concat(dfs, ignore_index=True)
        raise FileNotFoundError(f"未找到数据: {self.dataset_path}")
    
    def _detect_columns(self):
        # CALVIN 特定的列名
        self.base_image_col = find_column(self.data, [
            'observation.images.base_0_rgb', 
            'observation.images.rgb_static',
            'rgb_static'
        ])
        self.left_wrist_col = find_column(self.data, [
            'observation.images.left_wrist_0_rgb',
            'observation.images.rgb_gripper',
            'rgb_gripper'
        ])
        self.right_wrist_col = find_column(self.data, [
            'observation.images.right_wrist_0_rgb'
        ])
        self.state_col = find_column(self.data, [
            'observation.state', 
            'robot_obs', 
            'state'
        ])
        self.action_col = find_column(self.data, [
            'actions', 
            'action', 
            'rel_actions'
        ])
        self.episode_col = find_column(self.data, [
            'episode_index', 
            'episode'
        ])
        # 任务索引列
        self.task_index_col = find_column(self.data, [
            'task_index',
            'task_idx',
        ])
        
        if self.verbose:
            print(f"\n检测到的列:")
            print(f"  base_image: {self.base_image_col}")
            print(f"  left_wrist: {self.left_wrist_col}")
            print(f"  right_wrist: {self.right_wrist_col}")
            print(f"  state: {self.state_col}")
            print(f"  action: {self.action_col}")
            print(f"  episode: {self.episode_col}")
            print(f"  task_index: {self.task_index_col}")
    
    def _build_episodes(self, max_episodes):
        if not self.episode_col or self.episode_col not in self.data.columns:
            return [{'episode_index': 0, 'start': 0, 'end': len(self.data) - 1}]
        
        episodes = []
        unique_eps = sorted(self.data[self.episode_col].unique())
        if max_episodes:
            unique_eps = unique_eps[:max_episodes]
        
        for ep_idx in unique_eps:
            indices = self.data.index[self.data[self.episode_col] == ep_idx]
            episodes.append({
                'episode_index': ep_idx, 
                'start': indices.min(), 
                'end': indices.max()
            })
        return episodes
    
    def get_task_description(self, task_index: Any) -> str:
        """
        根据 task_index 获取任务描述
        """
        if task_index is None:
            return "perform the task"
        
        try:
            idx = int(task_index)
            if idx in self.task_map:
                return self.task_map[idx]
            else:
                return f"perform task {idx}"
        except (ValueError, TypeError):
            return "perform the task"
    
    def get_sample(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本，格式化为 CALVIN policy 期望的输入
        """
        row = self.data.iloc[idx]
        
        # 构建 images 字典
        images = {}
        
        if self.base_image_col:
            images["base_0_rgb"] = parse_image(row[self.base_image_col], self.dataset_path)
        else:
            images["base_0_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.left_wrist_col:
            images["left_wrist_0_rgb"] = parse_image(row[self.left_wrist_col], self.dataset_path)
        else:
            images["left_wrist_0_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.right_wrist_col:
            images["right_wrist_0_rgb"] = parse_image(row[self.right_wrist_col], self.dataset_path)
        else:
            images["right_wrist_0_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 状态
        if self.state_col:
            state = parse_array(row[self.state_col], 32)
        else:
            state = np.zeros(8, dtype=np.float32)
        
        # 任务提示 - 从 task_index 获取语言指令
        prompt = "perform the task"
        if self.task_index_col and self.task_index_col in self.data.columns:
            task_idx = row[self.task_index_col]
            prompt = self.get_task_description(task_idx)
        
        # 构建返回字典
        sample = {
            "images": images,
            "state": state,
            "prompt": prompt,
        }
        
        # 保存 GT action 用于评估
        if self.action_col:
            sample["gt_action"] = parse_array(row[self.action_col], 7)
        
        # 保存 task_index 用于分析
        if self.task_index_col and self.task_index_col in self.data.columns:
            sample["task_index"] = row[self.task_index_col]
        
        return sample
    
    def get_evaluation_samples(
        self, 
        action_horizon: int = 10, 
        num_samples: int = None,
        skip_interval: int = 1
    ) -> Tuple[List[Dict], np.ndarray]:
        """获取评估样本"""
        samples = []
        gt_trajectories = []
        
        for ep in tqdm(self.episodes, desc="提取样本", disable=not self.verbose):
            for start_idx in range(ep['start'], ep['end'] - action_horizon + 1, skip_interval):
                if start_idx + action_horizon > ep['end']:
                    break
                
                # 获取输入样本
                sample = self.get_sample(start_idx)
                
                # GT 轨迹
                gt_actions = []
                for i in range(action_horizon):
                    action = self.data.iloc[start_idx + i][self.action_col]
                    gt_actions.append(parse_array(action, 7))
                
                samples.append(sample)
                gt_trajectories.append(np.stack(gt_actions))
                
                if num_samples and len(samples) >= num_samples:
                    break
            
            if num_samples and len(samples) >= num_samples:
                break
        
        return samples, np.stack(gt_trajectories)


# ============================================================================
# 推理函数
# ============================================================================

def run_inference(
    policy,
    samples: List[Dict],
    action_horizon: int,
    action_dim: int = 7,
    show_progress: bool = True
) -> Tuple[np.ndarray, int]:
    """运行模型推理"""
    predictions = []
    num_errors = 0
    
    iterator = tqdm(samples, desc="推理") if show_progress else samples
    
    for sample in iterator:
        try:
            # 输入格式: {"images": {...}, "state": ..., "prompt": ...}
            input_dict = {
                "images": sample["images"],
                "state": sample["state"],
                "prompt": sample["prompt"],
            }
            
            # 调用推理
            result = policy.infer(input_dict)
            actions = np.array(result['actions'])
            
            # 调整长度
            if len(actions) >= action_horizon:
                actions = actions[:action_horizon]
            else:
                pad_len = action_horizon - len(actions)
                actions = np.concatenate([actions, np.tile(actions[-1:], (pad_len, 1))])
            
            # 调整维度
            if actions.shape[-1] < action_dim:
                pad = np.zeros((action_horizon, action_dim - actions.shape[-1]))
                actions = np.concatenate([actions, pad], axis=-1)
            else:
                actions = actions[..., :action_dim]
            
            predictions.append(actions)
            
        except Exception as e:
            num_errors += 1
            if num_errors <= 3:
                print(f"\n推理错误: {e}")
            predictions.append(np.zeros((action_horizon, action_dim), dtype=np.float32))
    
    return np.stack(predictions), num_errors


# ============================================================================
# 主评估函数
# ============================================================================

def run_evaluation(
    checkpoint_dir: str,
    dataset_path: str,
    config_name: str = "pi0_calvin_scratch",
    num_samples: int = 100,
    action_horizon: int = 10,
    skip_interval: int = 10,
    output_dir: str = None,
    output_filename: str = None,
    verbose: bool = True,
):
    """运行完整评估"""
    
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    
    if verbose:
        print("\n" + "=" * 70)
        print(" Pi0 CALVIN 真实推理评估 (支持语言指令)")
        print("=" * 70)
    
    # 1. 加载策略
    if verbose:
        print("\n【1】加载策略...")
    
    config = _config.get_config(config_name)
    
    if verbose:
        print(f"配置: {config_name}")
        print(f"  action_dim: {config.model.action_dim}")
        print(f"  action_horizon: {config.model.action_horizon}")
    
    try:
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        if verbose:
            print("✓ 策略加载成功")
    except Exception as e:
        print(f"❌ 策略加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 2. 加载数据集
    if verbose:
        print("\n【2】加载数据集...")
    
    dataset = CALVINEvalDataset(
        dataset_path, 
        max_episodes=num_samples * 2,
        verbose=verbose
    )
    
    # 3. 提取样本
    if verbose:
        print("\n【3】提取评估样本...")
    
    samples, gt_trajectories = dataset.get_evaluation_samples(
        action_horizon=action_horizon,
        num_samples=num_samples,
        skip_interval=skip_interval
    )
    
    if verbose:
        print(f"评估样本数: {len(samples)}")
        print(f"GT 轨迹形状: {gt_trajectories.shape}")
        
        # 打印一个样本的格式
        s = samples[0]
        print(f"\n样本格式:")
        print(f"  images.base_0_rgb: {s['images']['base_0_rgb'].shape}, {s['images']['base_0_rgb'].dtype}")
        print(f"  images.left_wrist_0_rgb: {s['images']['left_wrist_0_rgb'].shape}")
        print(f"  state: {s['state'].shape}, {s['state'].dtype}")
        print(f"  prompt: \"{s['prompt']}\"")
        
        # 统计任务分布
        if 'task_index' in samples[0]:
            task_counts = {}
            for samp in samples:
                t = samp.get('task_index', -1)
                task_counts[t] = task_counts.get(t, 0) + 1
            print(f"\n任务分布 (前5个):")
            for t, c in sorted(task_counts.items(), key=lambda x: -x[1])[:5]:
                desc = dataset.get_task_description(t)
                print(f"  [{t}] {desc}: {c} 样本")
    
    # 4. 推理
    if verbose:
        print("\n【4】运行模型推理...")
    
    action_dim = gt_trajectories.shape[-1]
    pred_trajectories, num_errors = run_inference(
        policy=policy,
        samples=samples,
        action_horizon=action_horizon,
        action_dim=action_dim,
        show_progress=verbose
    )
    
    if verbose:
        print(f"\n预测轨迹形状: {pred_trajectories.shape}")
        if num_errors > 0:
            print(f"⚠ {num_errors} 个样本推理出错")
        else:
            print("✓ 所有样本推理成功")
    
    # 5. 计算指标
    if verbose:
        print("\n【5】计算评估指标...")
    
    min_dim = min(pred_trajectories.shape[-1], gt_trajectories.shape[-1])
    metrics = evaluate_trajectory_np(
        pred_trajectories[..., :min_dim], 
        gt_trajectories[..., :min_dim]
    )
    
    # 6. 打印结果
    if verbose:
        print("\n" + "=" * 70)
        print(" 评估结果")
        print("=" * 70)
        
        print("\n📊 轨迹误差:")
        print(f"  MSE:  {metrics['traj_mse']:.6f}")
        print(f"  MAE:  {metrics['traj_mae']:.6f}")
        print(f"  RMSE: {metrics['traj_rmse']:.6f}")
        
        print("\n📍 位置误差:")
        print(f"  Position MSE:  {metrics['pos_mse']:.6f}")
        print(f"  Position RMSE: {metrics['pos_rmse_cm']:.2f} cm")
        print(f"  ADE: {metrics['ade_mean_cm']:.2f} cm (median: {metrics['ade_median_cm']:.2f})")
        print(f"  FDE: {metrics['fde_mean_cm']:.2f} cm (median: {metrics['fde_median_cm']:.2f})")
        
        if 'rot_error_mean_deg' in metrics:
            print("\n🔄 旋转误差:")
            print(f"  Mean:   {metrics['rot_error_mean_deg']:.2f}°")
            print(f"  Median: {metrics['rot_error_median_deg']:.2f}°")
            print(f"  Std:    {metrics['rot_error_std_deg']:.2f}°")
            
            print("\n✅ 成功率:")
            print(f"  SR (1cm, 2°):  {metrics['sr_1cm_2deg']*100:.2f}%")
            print(f"  SR (2cm, 5°):  {metrics['sr_2cm_5deg']*100:.2f}%  ← 推荐指标")
            print(f"  SR (3cm, 10°): {metrics['sr_3cm_10deg']*100:.2f}%")
            print(f"  SR (5cm, 15°): {metrics['sr_5cm_15deg']*100:.2f}%")
        
        print("\n📐 各维度 MSE:")
        for key in sorted([k for k in metrics.keys() if k.startswith('mse_')]):
            dim = key.replace('mse_', '')
            print(f"  {dim}: {metrics[key]:.6f}")
    
    # 7. 保存结果
    if output_dir is None:
        output_dir = Path(OPENPI_ROOT) / 'evaluation' / 'results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            checkpoint_name = Path(checkpoint_dir).parts[-2]
            step = Path(checkpoint_dir).name
        except:
            checkpoint_name = "model"
            step = "0"
        output_filename = f"eval_{checkpoint_name}_{step}_{timestamp}.json"
    
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_dir": str(checkpoint_dir),
            "dataset_path": str(dataset_path),
            "config_name": config_name,
            "num_samples": len(samples),
            "num_errors": num_errors,
            "action_horizon": action_horizon,
            "inference_mode": "real",
            "has_language_instructions": bool(dataset.task_map),
        },
        "metrics": metrics,
        "summary": {
            "trajectory_mse": metrics.get('traj_mse'),
            "position_mse": metrics.get('pos_mse'),
            "fde_cm": metrics.get('fde_mean_cm'),
            "ade_cm": metrics.get('ade_mean_cm'),
            "rotation_error_deg": metrics.get('rot_error_mean_deg'),
            "success_rate_2cm_5deg": metrics.get('sr_2cm_5deg'),
        }
    }
    
    output_path = output_dir / output_filename
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    if verbose:
        print(f"\n✓ 结果已保存到: {output_path}")
    
    # 创建最新结果链接
    latest_path = output_dir / "latest_eval.json"
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(output_path.name)
    except:
        import shutil
        shutil.copy(output_path, latest_path)
    
    if verbose:
        print("=" * 70)
    
    return metrics


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pi0 CALVIN 评估 (支持语言指令)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='检查点目录')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--config_name', type=str, default='pi0_calvin_scratch',
                        help='配置名称')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='评估样本数')
    parser.add_argument('--action_horizon', type=int, default=10,
                        help='动作序列长度')
    parser.add_argument('--skip_interval', type=int, default=10,
                        help='采样间隔')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--output_filename', type=str, default=None,
                        help='输出文件名')
    parser.add_argument('--quiet', action='store_true',
                        help='减少输出')
    
    args = parser.parse_args()
    
    run_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        dataset_path=args.dataset_path,
        config_name=args.config_name,
        num_samples=args.num_samples,
        action_horizon=args.action_horizon,
        skip_interval=args.skip_interval,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        verbose=not args.quiet,
    )