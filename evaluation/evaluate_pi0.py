#!/usr/bin/env python3
"""
Pi0 Model Evaluation Script v3.3 - Calvin / LeRobot compatible
==============================================================
- Eval now uses base_model.sample_actions(Observation) instead of policy.infer
- Avoids CALVINInputs / prompt / image_masks schema issues
- Fix: .cpu() branch bug when converting actions to numpy
- Memory friendly: streaming per-dimension MSE (no huge concatenation)
- Keep: norm_stats / assets handling from v3.1

Author: Claude 
Date: 2025-11-14

uv run evaluate_pi0.py \
    --config-name pi0_calvin_scratch \
    --checkpoint-dir /root/autodl-tmp/openpi/checkpoints/pi0_calvin_scratch/calvin_full/21000 \
    --dataset-path Coil1987121/calvin_lerobot_task_ABCD_D_validation \
    --batch-size 128
    --num_samples 3000

"""

import os
# 强制 JAX 使用平台默认 allocator，避免在 GPU/TPU 上出现碎片化导致的 OOM
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# 降低 TensorFlow/JAX 的日志级别，防止评估时刷屏
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import argparse
import json
import logging
import dataclasses
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import math
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# Configure logging
# 统一设置日志格式，便于在长时间评估中追踪关键事件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- NEW: load task_index -> prompt mapping from LeRobot meta/tasks.jsonl ----

def load_task_map(dataset_path: str) -> Optional[Dict[int, str]]:
    """
    尝试从 LeRobot 数据集目录中加载 meta/tasks.jsonl，建立
    task_index -> task 文本 的映射。
    
    支持两种情况：
      1) dataset_path 是 repo_id，例如 "Coil1987121/xxx"
      2) dataset_path 是本地路径，例如 "/root/autodl-tmp/huggingface/lerobot/Coil.../"
    """
    candidates = []

    # 情况 1：dataset_path 本身就是一个目录
    p = Path(dataset_path)
    # Path.exists 用于判断 dataset_path 是否为本地目录
    if p.exists():
        candidates.append(p / "meta" / "tasks.jsonl")

    # 情况 2：通过 HF_LEROBOT_HOME / repo_id 来找
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        # 将 HF 缓存根目录与 repo_id 拼出 meta/tasks.jsonl 的绝对路径
        candidates.append(HF_LEROBOT_HOME / dataset_path / "meta" / "tasks.jsonl")
    except Exception:
        pass

    task_file = None
    for c in candidates:
        if c.exists():
            task_file = c
            break

    if task_file is None:
        logger.warning(
            f"Could not find meta/tasks.jsonl for dataset_path={dataset_path}. "
            f"Prompt will fall back to default string."
        )
        return None

    logger.info(f"Loading task map from: {task_file}")
    task_map: Dict[int, str] = {}
    with open(task_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # json.loads 将每一行 JSONL 解析成 dict
                obj = json.loads(line)
                idx = int(obj["task_index"])
                txt = str(obj["task"])
                task_map[idx] = txt
            except Exception as e:
                logger.warning(f"Failed to parse line in tasks.jsonl: {line} ({e})")

    logger.info(f"Loaded {len(task_map)} tasks from tasks.jsonl")
    return task_map if task_map else None


def setup_environment():
    """Setup necessary environment variables and imports."""
    try:
        # 延迟导入 OpenPI 模块，避免在未安装 openpi 时立即报错
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.training import data_loader as _data_loader
        import openpi.training.sharding as sharding

        return config, policy_config, _data_loader, sharding
    except ImportError as e:
        logger.error(f"Failed to import openpi modules: {e}")
        logger.error("Please ensure openpi is installed: uv pip install -e .")
        sys.exit(1)


def extract_dataset_id_from_repo_id(repo_id: str) -> str:
    """Extract the dataset ID from a HuggingFace repo_id or local path."""
    if "/" in repo_id:
        return repo_id.split("/")[-1]
    return repo_id


def load_pi0_model_and_config(config_name: str, checkpoint_dir: str):
    """Load a trained pi0 model from checkpoint along with its training config."""
    logger.info(f"Loading config: {config_name}")
    logger.info(f"Loading checkpoint from: {checkpoint_dir}")

    config_module, policy_config_module, _, _ = setup_environment()

    try:
        # get_config 是 OpenPI 的配置工厂，按照名称加载 dataclass config
        train_config = config_module.get_config(config_name)
        # create_trained_policy 会从 checkpoint_dir 读取权重并构造 Policy 对象
        policy = policy_config_module.create_trained_policy(
            train_config, checkpoint_dir
        )
        logger.info("✓ Model loaded successfully!")
        return policy, train_config

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def find_norm_stats_in_assets(
    config_name: str, training_repo_id: str = None
) -> Optional[Path]:
    """
    Search for norm_stats.json in the standard OpenPI assets structure.

    OpenPI structure:
        assets/
          {config_name}/          # <- asset_id
            {username}/
              {dataset_id}/
                norm_stats.json
    """
    logger.info("Searching for norm_stats.json in OpenPI assets structure...")

    # Standard OpenPI assets locations
    assets_bases = [
        Path("/root/autodl-tmp/openpi/assets"),
        Path("./assets"),
        Path("assets"),
    ]

    found_paths = []

    for assets_base in assets_bases:
        # 逐个遍历候选根目录，拼出 config 专属子目录
        config_assets_dir = assets_base / config_name

        if config_assets_dir.exists():
            logger.info(f"Found config assets directory: {config_assets_dir}")
            # rglob 能递归扫描所有 username/dataset 子目录下的 norm_stats.json
            norm_stats_files = list(config_assets_dir.rglob("norm_stats.json"))
            found_paths.extend(norm_stats_files)

    if not found_paths:
        logger.warning(f"No norm_stats.json found for config: {config_name}")
        return None

    # If we have training_repo_id, try to match it
    if training_repo_id:
        training_dataset_id = extract_dataset_id_from_repo_id(training_repo_id)
        logger.info(
            f"Looking for norm_stats matching training dataset: {training_dataset_id}"
        )

        for path in found_paths:
            # path.parent.name 对应 dataset_id，拿来与训练 repo 匹配
            dataset_dir_name = path.parent.name
            if (
                training_dataset_id in dataset_dir_name
                or dataset_dir_name in training_dataset_id
            ):
                logger.info(f"✓ Found matching norm_stats: {path}")
                return path

    if len(found_paths) > 0:
        logger.warning(f"Using first found norm_stats: {found_paths[0]}")
        logger.warning("If incorrect, specify --norm-stats-path manually")
        return found_paths[0]

    return None


def create_assets_config_from_training(
    train_config, config_name: str
) -> Optional["AssetsConfig"]:
    """Create AssetsConfig by checking training config or searching assets."""
    # AssetsConfig 是训练/推理统一的资源描述 dataclass
    from openpi.training.config import AssetsConfig

    # FIRST: Check if training config already has assets
    if hasattr(train_config.data, "assets") and train_config.data.assets is not None:
        logger.info("✓ Using assets from training config:")
        logger.info(f"  assets_dir: {train_config.data.assets.assets_dir}")
        logger.info(f"  asset_id: {train_config.data.assets.asset_id}")
        return train_config.data.assets

    # SECOND: Try to find norm_stats and construct AssetsConfig
    logger.info("No assets in training config, searching for norm_stats...")

    training_repo_id = None
    if hasattr(train_config.data, "repo_id"):
        training_repo_id = train_config.data.repo_id
        logger.info(f"Training dataset repo_id: {training_repo_id}")

    # 调用上面的文件系统扫描，尝试找到 norm_stats.json 的真实位置
    norm_stats_path = find_norm_stats_in_assets(config_name, training_repo_id)

    if norm_stats_path is None:
        logger.warning("Could not find norm_stats.json in standard locations")
        return None

    # Construct AssetsConfig from the found norm_stats.json path
    # Path structure: assets_dir/asset_id/username/dataset_id/norm_stats.json
    dataset_dir = norm_stats_path.parent  # dataset_id
    username_dir = dataset_dir.parent  # username
    asset_id_dir = username_dir.parent  # asset_id
    assets_dir = asset_id_dir.parent  # assets base directory

    asset_id = asset_id_dir.name

    # 通过 AssetsConfig 传回 assets_dir/asset_id，供 data_loader 自动解析 norm_stats
    assets_config = AssetsConfig(
        assets_dir=str(assets_dir),
        asset_id=asset_id,
    )

    logger.info("✓ Created AssetsConfig:")
    logger.info(f"  assets_dir: {assets_config.assets_dir}")
    logger.info(f"  asset_id: {assets_config.asset_id}")
    logger.info(f"  norm_stats found at: {norm_stats_path}")

    return assets_config


def create_validation_data_loader(
    train_config,
    val_dataset_path: str,
    batch_size: int = 1,
    skip_norm_stats: bool = False,
    explicit_norm_stats_path: Optional[Path] = None
):
    """Create a data loader for validation with OpenPI-compliant norm stats handling.

    注意：这里的 data_loader 是单机版本（sharding=None），
    方便直接用 base_model.sample_actions 做推理。
    """
    # 复用训练阶段的数据加载实现，保持预处理完全一致
    _, _, _data_loader, _sharding = setup_environment()
    # AssetsConfig 提供 norm_stats 等资源的信息
    from openpi.training.config import AssetsConfig

    logger.info(f"Creating validation data loader for: {val_dataset_path}")
    
    # 基于训练 data config 复制一份，用新的 repo_id
    # dataclasses.replace 允许在不修改原 config 的前提下，替换 repo_id 指向验证集
    val_data_config = dataclasses.replace(
        train_config.data,
        repo_id=val_dataset_path,
    )
    
    # 处理 norm_stats / assets
    if not skip_norm_stats:
        assets_config = None
        
        # Option 1: 显式给了 norm_stats_path
        if explicit_norm_stats_path and explicit_norm_stats_path.exists():
            logger.info(f"Using explicit norm_stats path: {explicit_norm_stats_path}")
            
            # 规范路径结构：assets_dir/asset_id/username/dataset_id/norm_stats.json
            dataset_dir = explicit_norm_stats_path.parent      # dataset_id directory
            username_dir = dataset_dir.parent                  # username directory  
            asset_id_dir = username_dir.parent                 # asset_id directory
            assets_dir = asset_id_dir.parent                   # assets base directory
            
            logger.info("Path analysis:")
            logger.info(f"  norm_stats: {explicit_norm_stats_path.name}")
            logger.info(f"  dataset_id: {dataset_dir.name}")
            logger.info(f"  username: {username_dir.name}")
            logger.info(f"  asset_id: {asset_id_dir.name}")
            logger.info(f"  assets_dir: {assets_dir}")
            
            # AssetsConfig 需要 assets_dir + asset_id，剩余路径由 data loader 推断
            assets_config = AssetsConfig(
                assets_dir=str(assets_dir),
                asset_id=asset_id_dir.name,
            )
            
            logger.info("✓ Created AssetsConfig from explicit path:")
            logger.info(f"  assets_dir: {assets_config.assets_dir}")
            logger.info(f"  asset_id: {assets_config.asset_id}")
        
        # Option 2: 从训练 config / assets 目录里自动找
        else:
            # 如果没有手动指定，就复用训练配置或自动搜索到的 norm_stats
            assets_config = create_assets_config_from_training(
                train_config,
                train_config.name
            )
        
        # 应用 assets_config
        if assets_config is not None:
            val_data_config = dataclasses.replace(
                val_data_config,
                assets=assets_config
            )
            logger.info("✓ Applied assets config to validation data loader")
        else:
            logger.warning("Could not find norm_stats.json")
            logger.warning("Falling back to skip_norm_stats mode")
            skip_norm_stats = True
    
    # 最终的验证 config
    # 再复制整个训练 config，只改数据配置和 batch size，即可重用训练脚本里的 data pipeline
    val_config = dataclasses.replace(
        train_config,
        data=val_data_config,
        batch_size=batch_size,
    )
    
    try:
        # ★ 关键：验证阶段用 sharding=None，拿到单机版的 Observation / Actions
        # create_data_loader 是 OpenPI 数据接口，sharding=None 表示拿到单机批次
        data_loader = _data_loader.create_data_loader(
            val_config,
            sharding=None,
            shuffle=False,
            skip_norm_stats=skip_norm_stats,
        )
        
        logger.info("✓ Validation data loader created successfully")
        
    except Exception as e:
        error_msg = str(e)
        
        if "Normalization stats not found" in error_msg or "Norm stats file not found" in error_msg:
            logger.error("")
            logger.error("="*70)
            logger.error("FAILED TO LOAD NORMALIZATION STATS")
            logger.error("="*70)
            logger.error("")
            logger.error("The norm_stats.json file could not be found.")
            logger.error("")
            logger.error("Expected structure:")
            logger.error("  assets/{config_name}/{username}/{dataset_id}/norm_stats.json")
            logger.error("")
            logger.error("Solutions:")
            logger.error("")
            logger.error("1. Generate norm_stats for training dataset:")
            logger.error(f"   uv run scripts/compute_norm_stats.py --config-name={train_config.name}")
            logger.error("")
            logger.error("2. Skip normalization (for quick testing only):")
            logger.error("   Add --skip-norm-stats flag")
            logger.error("")
            logger.error("3. Specify norm_stats path manually:")
            logger.error("   Add --norm-stats-path /path/to/norm_stats.json")
            logger.error("")
            logger.error("="*70)
            raise
        else:
            raise
    
    # 为了不改 main() 的解包逻辑，这里仍然返回一个 mesh（设为 None）
    mesh = None
    return data_loader, mesh
    

# === NEW: helper to convert Observation -> dict for calvin_policy ==================
# （目前没用到，保留以防未来需要 policy.infer 路径）

def _observation_to_policy_input(obs: Any,
                                 task_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """
    将训练时的 Observation 对象转换成 policy.infer 需要的 dict 格式。
    当前 compute_trajectory_mse 直接用 base_model.sample_actions，不走这个路径。
    """
    if isinstance(obs, dict):
        data = dict(obs)
    elif hasattr(obs, "_asdict"):
        # 命名元组可以通过 _asdict() 直接转 dict
        data = obs._asdict()
    elif getattr(obs, "__dataclass_fields__", None) is not None or dataclasses.is_dataclass(obs):
        # dataclasses.asdict 会递归拷贝 Observation dataclass
        data = dataclasses.asdict(obs)
    elif hasattr(obs, "__dict__"):
        data = {k: getattr(obs, k) for k in vars(obs)}
    else:
        return obs

    if "images" not in data and "observation" in data and isinstance(data["observation"], dict):
        inner = data["observation"]
        if "images" in inner and "images" not in data:
            data["images"] = inner["images"]
        if "state" in inner and "state" not in data:
            data["state"] = inner["state"]
        if "actions" in inner and "actions" not in data:
            data["actions"] = inner["actions"]
        for k in ("prompt", "task", "language_instruction", "task_index"):
            if k in inner and k not in data:
                data[k] = inner[k]

    def _is_nonempty_str(x):
        return isinstance(x, str) and x.strip() != ""

    prompt = None

    if _is_nonempty_str(data.get("prompt", None)):
        prompt = data["prompt"]

    if prompt is None:
        for key in ("task", "language_instruction", "instruction", "text"):
            val = data.get(key, None)
            if _is_nonempty_str(val):
                prompt = val
                break

    if prompt is None:
        for container_key in ("info", "meta"):
            inner = data.get(container_key, None)
            if isinstance(inner, dict):
                for key in ("prompt", "task", "language_instruction"):
                    val = inner.get(key, None)
                    if _is_nonempty_str(val):
                        prompt = val
                        break
            if prompt is not None:
                break

    def _extract_task_index(d: dict) -> Optional[int]:
        for key in ("task_index", "task_id", "task"):
            val = d.get(key, None)
            if isinstance(val, (int, np.integer)):
                return int(val)
        for container_key in ("info", "meta", "observation"):
            inner = d.get(container_key, None)
            if isinstance(inner, dict):
                idx = _extract_task_index(inner)
                if idx is not None:
                    return idx
        return None

    if prompt is None and task_map is not None:
        idx = _extract_task_index(data)
        if idx is not None and idx in task_map:
            prompt = task_map[idx]
            logger.debug(f"Using prompt from task_map[{idx}]: {prompt}")

    if prompt is None:
        prompt = "Follow the task instruction."
        logger.debug("No prompt / task / language_instruction / task_index found; using default prompt.")

    data["prompt"] = prompt

    if "images" not in data:
        raise ValueError("Observation-to-dict conversion failed: no 'images' field found.")
    if "state" not in data:
        raise ValueError("Observation-to-dict conversion failed: no 'state' field found.")

    return data


def compute_trajectory_mse(
    policy,
    data_loader,
    num_samples: Optional[int] = None,
    strict: bool = False,
) -> Dict[str, float]:
    """
    使用底层 base_model.sample_actions 计算 Trajectory MSE。

    和训练脚本对齐：
    - data_loader 直接给 (_model.Observation, Actions)
    - 我们用 sample_actions(rng, observation) 得到预测动作
    - 与 GT actions 做 MSE
    """
    logger.info("Computing Trajectory MSE using base_model.sample_actions...")
    logger.info(f"Target samples: {num_samples if num_samples else 'all'}")
    
    # 从 policy 里拿到底层模型
    # policy._model 是训练时的 flax/haiku 模型对象
    base_model = getattr(policy, "_model", None)
    if base_model is None:
        logger.error("Policy object does not have attribute '_model'.")
        logger.error("This evaluation script assumes openpi.policies.Policy with _model field.")
        raise RuntimeError("Cannot access underlying model from policy.")
    
    # 使用 JAX 的 PRNGKey 保持与训练一致的随机性
    rng = jax.random.PRNGKey(0)
    
    errors: list[float] = []
    per_dim_sse: Optional[np.ndarray] = None  # sum of squared error per dim
    per_dim_count: int = 0                    # total count per dim (B * T 累加)
    
    num_processed = 0
    num_errors = 0
    
    # tqdm 提供推理进度条，方便评估长任务
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Running inference")):
        try:
            # 解包 batch：和 train_step 一样，期望是 (Observation, Actions)
            if isinstance(batch, tuple) and len(batch) == 2:
                observation, actions = batch
            elif hasattr(batch, "__getitem__") and len(batch) == 2:
                observation, actions = batch[0], batch[1]
            else:
                if num_errors < 5:
                    logger.warning(f"Unexpected batch format at {batch_idx}: {type(batch)}")
                num_errors += 1
                continue
            
            # 采样用的随机数
            # 每个 batch 拆分 RNG，保证 sample_actions 接口获得独立随机数
            rng, sample_rng = jax.random.split(rng)
            
            # ★ 核心：直接在 Observation 上跑模型
            # 底层 API：直接对 Observation 调用 sample_actions 进行自回归采样
            pred_actions_jax = base_model.sample_actions(sample_rng, observation)
            
            # 转成 numpy（避免多余拷贝，用 asarray）
            # np.asarray 尽量零拷贝地把 JAX/DeviceArray 转成 numpy
            pred_actions = np.asarray(pred_actions_jax)
            gt_actions = np.asarray(actions)
            
            # 统一成 [B, T, D] 格式（如果是 [B, D]，就当 T=1）
            if pred_actions.ndim == 2:
                pred_actions = pred_actions[:, None, :]
            if gt_actions.ndim == 2:
                gt_actions = gt_actions[:, None, :]
            
            # 对齐时间步 / 维度（如果略有不一致就裁到最小公共形状）
            if pred_actions.shape != gt_actions.shape:
                if pred_actions.shape[0] != gt_actions.shape[0]:
                    if num_errors < 5:
                        logger.warning(
                            f"Batch {batch_idx}: batch size mismatch, "
                            f"pred {pred_actions.shape}, gt {gt_actions.shape}"
                        )
                    num_errors += 1
                    continue
                
                min_T = min(pred_actions.shape[1], gt_actions.shape[1])
                min_D = min(pred_actions.shape[2], gt_actions.shape[2])
                
                if min_T == 0 or min_D == 0:
                    if num_errors < 5:
                        logger.warning(
                            f"Batch {batch_idx}: zero-length along T or D axis, "
                            f"pred {pred_actions.shape}, gt {gt_actions.shape}"
                        )
                    num_errors += 1
                    continue
                
                if num_errors < 5:
                    logger.warning(
                        f"Batch {batch_idx}: shape mismatch, cropping to "
                        f"[B, {min_T}, {min_D}] from pred {pred_actions.shape}, gt {gt_actions.shape}"
                    )
                
                pred_actions = pred_actions[:, :min_T, :min_D]
                gt_actions = gt_actions[:, :min_T, :min_D]
            
            # 现在 pred_actions 和 gt_actions 都是 [B, T, D]
            sq_err = (pred_actions - gt_actions) ** 2  # [B, T, D]
            # np.mean 直接沿 batch/time 维度求均值，得到 trajectory 级别的 MSE
            batch_mse = np.mean(sq_err, axis=(1, 2))   # 每个样本一个 MSE
            
            # 累积总体 trajectory MSE
            errors.extend(batch_mse.tolist())
            
            # 流式累积 per-dimension MSE：在 batch + time 上求和
            sse = sq_err.sum(axis=(0, 1))            # [D]
            count = sq_err.shape[0] * sq_err.shape[1]
            if per_dim_sse is None:
                per_dim_sse = sse
                per_dim_count = count
            else:
                per_dim_sse = per_dim_sse + sse
                per_dim_count = per_dim_count + count
            
            num_processed += pred_actions.shape[0]
            
            if num_samples is not None and num_processed >= num_samples:
                # 如果只想评估前 num_samples 个样本，就提前退出
                break
        
        except Exception as e:
            if num_errors < 5 or strict:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                logger.error(f"Batch type: {type(batch)}")
                if hasattr(batch, '__len__'):
                    logger.error(f"Batch length: {len(batch)}")
                import traceback
                logger.error(traceback.format_exc())
            elif num_errors == 5:
                logger.warning("Suppressing further detailed error messages...")
            
            if strict:
                logger.error("Strict mode enabled, failing immediately on error")
                raise
            
            num_errors += 1
            
            # 太多错误就停
            if num_errors > max(100, num_processed * 0.5):
                logger.error(
                    f"Too many errors encountered ({num_errors} errors, {num_processed} processed). "
                    "Stopping evaluation early."
                )
                break
            
            continue
    
    if len(errors) == 0:
        logger.error("No valid samples were processed!")
        logger.error(f"Total samples processed: {num_processed}")
        logger.error(f"Total errors: {num_errors}")
        logger.error("")
        logger.error("Possible causes:")
        logger.error("1. Dataset format is incompatible with the model")
        logger.error("2. Data loading pipeline has errors")
        logger.error("3. Incorrect norm_stats or data transforms")
        logger.error("")
        logger.error("Note: this path uses the same Observation structure as training.")
        return {}
    
    logger.info(f"✓ Successfully evaluated {len(errors)} samples")
    if num_errors > 0:
        logger.warning(
            f"Encountered {num_errors} errors "
            f"({num_errors / (num_processed + num_errors) * 100:.1f}% error rate)"
        )
    
    # 汇总 list -> numpy array 便于做统计指标
    errors_arr = np.asarray(errors, dtype=np.float32)
    
    results = {
        "trajectory_mse": float(errors_arr.mean()),
        "trajectory_mse_std": float(errors_arr.std()),
        "trajectory_mse_median": float(np.median(errors_arr)),
        "trajectory_mse_min": float(errors_arr.min()),
        "trajectory_mse_max": float(errors_arr.max()),
        "num_samples_evaluated": int(len(errors)),
        "num_failed_samples": int(num_errors),
    }
    
    # 流式 per-dimension MSE（不再拼巨大数组）
    if per_dim_sse is not None and per_dim_count > 0:
        per_dim_mse = per_dim_sse / float(per_dim_count)  # [D]
        results["per_dimension_mse"] = per_dim_mse.tolist()
        logger.info(f"Per-dimension MSE: {per_dim_mse}")
    else:
        logger.warning("Could not compute per-dimension MSE (no valid per-dim stats).")
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # json.dump 将字典持久化，indent=2 方便人工检查
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {output_path}")


def main():
    # argparse 用于声明 CLI 接口，RawDescriptionHelpFormatter 保留多行描述
    parser = argparse.ArgumentParser(
        description="Evaluate pi0 model (OpenPI + LeRobot CALVIN compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config-name", type=str, required=True, help="Name of the openpi config"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="HF repo_id or local path for validation dataset",
    )
    parser.add_argument(
        "--norm-stats-path", type=str, default=None, help="Path to norm_stats.json"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save results (JSON)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--skip-norm-stats",
        action="store_true",
        help="Skip normalization (for debugging)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on any error (default: skip errors and continue)",
    )

    args = parser.parse_args()

    # Print header
    logger.info("=" * 70)
    logger.info("PI0 MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config_name}")
    logger.info(f"Checkpoint: {args.checkpoint_dir}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Skip norm stats: {args.skip_norm_stats}")
    if args.norm_stats_path:
        logger.info(f"Norm stats path: {args.norm_stats_path}")
    logger.info("=" * 70)

    # Auto-generate output path
    if args.output_path is None:
        args.output_path = f"./eval_results_{args.config_name}.json"
        logger.info(f"Output path: {args.output_path}")

    # Validate paths
    # Path.exists 校验 checkpoint 目录是否存在，避免浪费时间跑加载
    if not Path(args.checkpoint_dir).exists():
        logger.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    # Load model
    logger.info("\nLoading model...")
    # load_pi0_model_and_config 会调用 OpenPI 配置/策略工厂
    policy, train_config = load_pi0_model_and_config(
        args.config_name, args.checkpoint_dir
    )

    # Create data loader
    logger.info("\nCreating validation data loader...")

    explicit_norm_stats_path = Path(args.norm_stats_path) if args.norm_stats_path else None

    data_loader, mesh = create_validation_data_loader(
        train_config,
        args.dataset_path,
        batch_size=args.batch_size,
        skip_norm_stats=args.skip_norm_stats,
        explicit_norm_stats_path=explicit_norm_stats_path,
    )

    # === 用 LeRobotDataset 统计总样本数（可选信息） ===
    try:
        # LeRobotDataset 负责从 HF Hub 或本地缓存读取序列长度
        eval_dataset = LeRobotDataset(repo_id=args.dataset_path)
        num_samples_total = len(eval_dataset)
        num_iters = math.ceil(num_samples_total / args.batch_size)

        logger.info(f"Total validation samples (frames): {num_samples_total}")
        logger.info(f"With batch_size={args.batch_size}, this is ~{num_iters} steps (it)")
    except Exception as e:
        logger.warning(f"Could not load LeRobotDataset to count samples: {e}")
    # ===========================================

    # Compute metrics
    logger.info("\nStarting evaluation...")
    # compute_trajectory_mse 用底层 base_model.sample_actions 跑推理
    results = compute_trajectory_mse(
        policy=policy,
        data_loader=data_loader,
        num_samples=args.num_samples,
        strict=args.strict,
    )

    if not results:
        logger.error("Evaluation failed")
        sys.exit(1)

    # Add metadata
    results["metadata"] = {
        "config_name": args.config_name,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset_path": args.dataset_path,
        "num_samples_requested": args.num_samples,
        "batch_size": args.batch_size,
        "skip_norm_stats": args.skip_norm_stats,
        "norm_stats_path": str(explicit_norm_stats_path)
        if explicit_norm_stats_path
        else None,
    }

    # Print results
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    for key, value in results.items():
        if key not in ["metadata", "per_dimension_mse"]:
            logger.info(f"{key:30s}: {value}")
    if "per_dimension_mse" in results:
        logger.info(f"per_dimension_mse: {results['per_dimension_mse']}")
    logger.info("=" * 70)

    # Save results
    # 评估完成后将指标写到 JSON 方便后续分析
    save_results(results, args.output_path)

    logger.info("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
