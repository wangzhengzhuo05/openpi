---
description: 'VLA dataset processing, validation, and preparation standards for robotic learning datasets'
applyTo: '**/*dataset*.py, **/*data*.py, **/process*.py, **/preprocess*.py'
---

# Dataset Processing and Preparation

Instructions for processing Vision-Language-Action datasets for robotic learning, including data validation, loading pipelines, and preprocessing workflows.

## General Instructions

### Dataset Selection and Assessment

- Identify the VLA dataset source (Open X-Embodiment, Bridge V2, CALVIN, RLBench, etc.)
- Determine task domain: manipulation, navigation, mobile manipulation, or multi-task
- Specify data format requirements: RLDS, HDF5, or custom formats
- Assess available storage and computational resources

### Data Format Validation Protocol

**CRITICAL: Always verify data formats before coding**

| Validation Step | Action Required | Success Criteria |
|----------------|-----------------|------------------|
| Input format | Confirm shape, dtype, structure | Exact specifications documented |
| Output format | Define expected format | User confirmation obtained |
| Preprocessing | List all transformations | Pipeline steps clearly defined |
| Domain conventions | Check standards | No ambiguity remains |

**When data format is unclear:**

1. **ASK clarifying questions**:
   - "What is the exact shape/format of your input data?"
   - "Can you show me a sample of your data structure?"
   - "What format do you expect the output to be in?"

2. **Provide diagnostic assistance**:
   - Suggest inspection commands: `print(data.shape)`, `data.info()`
   - Offer to write data exploration code
   - Explain common data formats in the domain

3. **Request reference materials**:
   - Ask for example code or reference implementations
   - Request sample input/output pairs
   - Suggest similar implementations from known libraries

**Only proceed with coding after:**
- Data formats are clearly understood and documented
- User has confirmed the requirements
- No ambiguity remains about data structure

## Best Practices

### Data Integrity and Transparency

| Practice | Implementation | Anti-Pattern |
|----------|----------------|---------------|
| Random data warning | Explicitly warn when generating synthetic data | Silently using random data |
| Transformation logging | Log all parameters and provide before/after stats | Silent data modifications |
| Data validation | Validate shapes and ranges after transformations | Assuming data is correct |
| Error reporting | Warn if data loss occurs with counts | Silently dropping samples |

### Code Examples

**✅ Good - Transparent Data Processing:**

```python
# Key principles: Log transformations, return stats, show progress

def process_trajectory(trajectory, target_length=100):
    """处理轨迹数据，包含裁剪/填充"""
    original_length = len(trajectory)
    
    # Log transformation
    logger.info(f"⚠️ Resizing trajectory: {original_length} → {target_length}")
    
    if original_length > target_length:
        processed = trajectory[:target_length]
        stats = {'operation': 'truncated', 'samples_removed': original_length - target_length}
        logger.warning(f"Truncated {stats['samples_removed']} samples")
    elif original_length < target_length:
        padding = np.zeros((target_length - original_length, trajectory.shape[1]))
        processed = np.vstack([trajectory, padding])
        stats = {'operation': 'padded', 'samples_added': target_length - original_length}
    else:
        processed = trajectory
        stats = {'operation': 'no_change'}
    
    # Log before/after stats
    logger.info(f"Before: shape={trajectory.shape}, range=[{trajectory.min():.3f}, {trajectory.max():.3f}]")
    logger.info(f"After: shape={processed.shape}, range=[{processed.min():.3f}, {processed.max():.3f}]")
    
    return processed, stats  # Return data with metadata
```

**❌ Bad - Silent Modifications:**

```python
def process_trajectory(trajectory):
    # No logging, no stats, no transparency
    if len(trajectory) > 100:
        trajectory = trajectory[:100]  # Silent truncation
    trajectory = (trajectory - trajectory.mean()) / trajectory.std()  # Silent normalization
    return trajectory  # No metadata about what changed
```

## Code Standards

### File Organization Requirements

**Multi-file tasks (≥2 files) MUST:**

1. Create dedicated folder with descriptive name
2. Include README.md with risk documentation
3. Follow standard structure:

```
dataset_task_name/
├── 1_validate_input.py    # 输入数据检查代码 (必须接受 --input_dir 参数)
├── 2_process.py           # 任务执行主代码 (必须接受 --input_dir 和 --output_dir 参数)
├── 3_validate_output.py   # 输出数据检查代码 (必须接受 --output_dir 参数)
├── run.sh                 # 可直接执行的完整脚本 (接受 input_dir 和 output_dir 参数)
├── README.md              # 必须: 任务说明、使用方法、风险说明
├── config.yaml            # 配置文件 (可选，推荐)
└── requirements.txt       # Python依赖 (可选)
```

**重要说明：**
- 所有 Python 脚本必须接受命令行参数指定输入输出路径
- run.sh/run.ps1 脚本必须接受路径参数：`bash run.sh <input_dir> <output_dir>`
- 在创建此文件夹前，必须先确认代码位置（用户指定或推荐后获得批准）

### Documentation Standards

| Element | Language | Format | Required |
|---------|----------|--------|----------|
| File header | English | Purpose, Usage, Dependencies | Yes |
| Function docstrings | Chinese | Args, Returns, Raises | Yes |
| Inline comments | Chinese | Explain WHY, not WHAT | As needed |
| README | Chinese | Full documentation | Multi-file tasks |

### Progress Tracking Requirements

**All time-consuming operations MUST show progress:**

```python
from tqdm import tqdm

for i in tqdm(range(total), desc="Processing dataset"):
    # Processing logic
    pass

# Output: Processing dataset: 1500/10000 (15%) - ETA: 5m 30s
```

## Common Patterns

### Pattern 1: Dataset Loading with Validation

```python
def load_and_validate_dataset(data_path: Path, expected_keys: list[str]) -> Dict:
    """加载并验证数据集"""
    logger.info(f"Loading dataset from {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # Validate keys exist
        missing = set(expected_keys) - set(f.keys())
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        data = {key: f[key][:] for key in expected_keys}
    
    # Log loaded data statistics
    for key, value in data.items():
        logger.info(f"Loaded '{key}': shape={value.shape}, dtype={value.dtype}")
    
    return data
```

### Pattern 2: Preprocessing Pipeline with Logging

```python
class PreprocessingPipeline:
    """数据预处理管道，带透明日志"""
    def __init__(self, steps: List[Callable]):
        self.steps = steps
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        logger.info(f"Starting pipeline with {len(self.steps)} steps")
        
        for i, step in enumerate(self.steps, 1):
            logger.info(f"Step {i}: {step.__name__}")
            before = f"mean={data.mean():.3f}, std={data.std():.3f}"
            data = step(data)
            after = f"mean={data.mean():.3f}, std={data.std():.3f}"
            logger.info(f"  Before: {before} | After: {after}")
        
        return data

# Usage
pipeline = PreprocessingPipeline([normalize, resize, augment])
result = pipeline(raw_data)
```

### Pattern 3: Safe Destructive Operations

```python
def delete_processed_cache(cache_dir: Path, require_confirmation: bool = True) -> bool:
    """安全删除缓存，带确认"""
    if not cache_dir.exists():
        return True
    
    # Calculate and warn about size
    total_size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*')) / (1024**2)
    logger.warning(f"⚠️ DESTRUCTIVE: Deleting {cache_dir} ({total_size_mb:.2f} MB)")
    
    # Require explicit confirmation
    if require_confirmation:
        if input("Type 'DELETE' to confirm: ") != 'DELETE':
            logger.info("Cancelled by user")
            return False
    
    shutil.rmtree(cache_dir)
    logger.info("Deleted successfully")
    return True
```

## Validation

### Pre-Processing Checks

| Check | Command | Expected Result |
|-------|---------|----------------|
| Data file exists | `Path(data_path).exists()` | True |
| Sufficient disk space | `shutil.disk_usage(path).free > required_gb * 1e9` | True |
| Required packages | `import h5py, numpy, tqdm` | No ImportError |

### Build and Test Commands

```bash
# Validate data processing script
python -m py_compile process_data.py

# Run with dry-run mode
python process_data.py --dry-run --input ./test_data

# Run full validation
python validate_data.py --input ./processed_data --check-all
```

### Quality Checklist

Before finalizing data processing code:

- [ ] Data formats validated and documented
- [ ] All transformations explicitly logged with parameters
- [ ] Progress tracking implemented for long operations
- [ ] README.md created with comprehensive risk warnings
- [ ] File organization follows standard structure
- [ ] Safe file operations with user confirmation
- [ ] Error handling covers edge cases
- [ ] Code passes linting (Black, Flake8)

## Risk Documentation Format (Required in README.md)

```markdown
## ⚠️ 风险说明 / Risk Warnings

### 🔴 Critical Risks
- **Data Overwrite**: This script will overwrite existing processed data in `./output/` directory
  - **Mitigation**: Backup original data before running
  - **Recovery**: Use `--restore` flag to recover from backup

### 🟡 Medium Risks
- **Disk Space**: Requires 50GB free space for processing
  - **Check**: Run `df -h` to verify available space
  - **Cleanup**: Use `--cleanup-temp` flag to remove intermediate files

### 🟢 Low Risks
- **Processing Time**: May take 2-4 hours for large datasets
  - **Monitoring**: Use `--verbose` flag to track progress
  - **Cancellation**: Use Ctrl+C to safely interrupt (state is checkpointed)

## 数据转换说明 / Data Transformations

| Transformation | Parameters | Effect |
|----------------|-----------|--------|
| Resize | target_length=100 | Truncates or pads trajectories to 100 steps |
| Normalize | mean=0, std=1 | Standardizes action values |
| Subsample | rate=10Hz | Reduces frame rate from original |
```
