"""
Merge two LeRobot-format datasets into a single dataset.

Usage:
    python merge_lerobot_datasets.py \
        --dataset1_repo Coil1987121/calvin_lerobot_task_ABCD_D_part1 \
        --dataset2_repo Coil1987121/calvin_lerobot_task_ABCD_D_part2 \
        --merged_repo Coil1987121/calvin_lerobot_task_ABCD_D_full
"""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import shutil
import tyro


def merge_datasets(dataset1_repo, dataset2_repo, merged_repo, push_to_hub=False):
    """
    Robust merge version:
      - If a dataset fails to load fully, tries to read its episode folders directly.
      - Skips broken or incomplete episodes automatically.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import json
    from pathlib import Path
    import warnings

    def try_load_dataset(repo_id: str):
        """Try loading dataset; fall back to manual episode folder read."""
        try:
            ds = LeRobotDataset(repo_id=repo_id)
            print(f"  ✓ Loaded via LeRobotDataset ({len(ds)} frames, {len(ds.episodes)} episodes)")
            return ds
        except Exception as e:
            print(f"  ⚠️ Dataset {repo_id} failed to load: {e}")
            # fallback: try direct episode folder read
            base_path = Path("/root/autodl-tmp/huggingface/lerobot") / repo_id
            episodes_dir = base_path / "episodes"
            if not episodes_dir.exists():
                print(f"  ❌ No local folder for {repo_id}, skipping entirely.")
                return None
            episodes = sorted(list(episodes_dir.glob("episode_*")))
            if not episodes:
                print(f"  ❌ No episode folders in {episodes_dir}, skipping.")
                return None
            print(f"  🧩 Fallback mode: found {len(episodes)} episode folders.")
            return {"repo_id": repo_id, "path": base_path, "episodes": episodes}

    # try load both datasets
    ds1 = try_load_dataset(dataset1_repo)
    ds2 = try_load_dataset(dataset2_repo)
    datasets = [ds for ds in [ds1, ds2] if ds is not None]

    if not datasets:
        raise RuntimeError("❌ No valid dataset could be loaded or recovered.")

    # use first dataset's schema if possible
    first_ds = datasets[0]
    if isinstance(first_ds, LeRobotDataset):
        base_features = first_ds.features
        base_fps = first_ds.fps
        base_robot = first_ds.robot_type
    else:
        # Fallback: guess
        base_features = {
            "observation.images.base_0_rgb": {"dtype": "image", "shape": (224, 224, 3)},
            "observation.state": {"dtype": "float32", "shape": (32,)},
            "actions": {"dtype": "float32", "shape": (7,)},
        }
        base_fps = 30
        base_robot = "franka"

    merged = LeRobotDataset.create(
        repo_id=merged_repo,
        robot_type=base_robot,
        fps=base_fps,
        features=base_features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    seen_ids = set()
    added, skipped = 0, 0

    for ds in datasets:
        if isinstance(ds, LeRobotDataset):
            for ep in ds.episodes:
                eid = getattr(ep, "episode_id", None)
                if eid in seen_ids:
                    skipped += 1
                    continue
                try:
                    merged.add_episode(ep)
                    seen_ids.add(eid)
                    added += 1
                except Exception as e:
                    warnings.warn(f"⚠️ Skipping bad episode {eid}: {e}")
                    skipped += 1
        else:
            for ep_dir in ds["episodes"]:
                eid = ep_dir.name
                if eid in seen_ids:
                    continue
                meta = ep_dir / "metadata.json"
                parquet = ep_dir / "data.parquet"
                if not parquet.exists():
                    skipped += 1
                    continue
                try:
                    merged.add_episode_from_disk(str(ep_dir))
                    seen_ids.add(eid)
                    added += 1
                except Exception as e:
                    warnings.warn(f"⚠️ Skipping raw episode {eid}: {e}")
                    skipped += 1

    merged.save()
    print(f"\n✅ Merged dataset saved to: ~/.cache/huggingface/lerobot/{merged_repo}")
    print(f"  Added episodes: {added}")
    print(f"  Skipped episodes: {skipped}")
    print(f"  Total frames: {len(merged)}")

    if push_to_hub:
        merged.push_to_hub(private=False, license="mit")


def main(
    dataset1_repo: str,
    dataset2_repo: str,
    merged_repo: str,
    push_to_hub: bool = False,
):
    merge_datasets(dataset1_repo, dataset2_repo, merged_repo, push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
