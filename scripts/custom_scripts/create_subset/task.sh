#!/bin/bash

DATA_DIR="/root/autodl-tmp/task_ABCD_D_processed/training"
STEP=3000
MAX=24000
SCRIPT="create_subset.py"

echo "Starting incremental subset creation..."
echo "Data dir: $DATA_DIR"
echo "Target max episodes: $MAX"
echo "Step: $STEP"
echo

current=$STEP

while [ $current -le $MAX ]; do
    if [ $current -eq $STEP ]; then
        # 第一轮，不需要 --resume
        echo "=== Running initial step: $current episodes ==="
        uv run "$SCRIPT" --data-dir "$DATA_DIR" --max_episodes "$current"
    else
        # 后续所有运行都需要 --resume
        echo "=== Running resume step: $current episodes ==="
        uv run "$SCRIPT" --data-dir "$DATA_DIR" --max_episodes "$current" --resume
    fi

    echo "=== Step $current done ==="
    echo

    current=$((current + STEP))
done

echo "All tasks finished!"
