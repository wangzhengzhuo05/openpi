"""CALVIN policy transforms for π₀ (Pi0) - LeRobot format."""

import dataclasses
from typing import ClassVar
import numpy as np
from openpi import transforms
import torch


def make_calvin_example() -> dict:
    """Creates a random input example for the CALVIN policy (π₀ format)."""
    return {
        "state": np.random.randn(32).astype(np.float32),  # 32D robot state
        "images": {
            "base_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "tokenized_prompt": np.random.randint(0, 100, size=(48,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones(48, dtype=bool),
    }


@dataclasses.dataclass(frozen=True)
class CALVINInputs(transforms.DataTransformFn):
    """Transforms LeRobot-format CALVIN data into π₀ model input format."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
    )

    @staticmethod
    def to_numpy_uint8(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return img

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]

        # Ensure all required cameras are present
        missing = set(self.EXPECTED_CAMERAS) - set(in_images)
        if missing:
            raise ValueError(f"Missing required cameras: {missing}")

        # ✅ Convert each image to numpy uint8 safely
        images = {
            "base_0_rgb": self.to_numpy_uint8(in_images["base_0_rgb"]),
            "left_wrist_0_rgb": self.to_numpy_uint8(in_images["left_wrist_0_rgb"]),
            "right_wrist_0_rgb": self.to_numpy_uint8(in_images["right_wrist_0_rgb"]),
        }

        # Mask right wrist (CALVIN only has 2 cameras)
        image_masks = {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False,  # duplicated view
        }

        # ✅ Safe dtype handling for state
        state = data["state"]
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        elif isinstance(state, torch.Tensor):
            state = state.float()
        else:
            state = np.array(state, dtype=np.float32)

        inputs = {
            "image": images,            # ✅ single key
            "image_mask": image_masks,  # ✅ single key
            "state": state,
        }

        # Optional text tokens
        if "tokenized_prompt" in data:
            inputs["tokenized_prompt"] = data["tokenized_prompt"]
        if "tokenized_prompt_mask" in data:
            inputs["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]

        # ✅ Handle both "action" and "actions" safely
        if "actions" in data:
            actions = data["actions"]
        elif "action" in data:
            actions = data["action"]
        else:
            actions = None

        if actions is not None:
            if isinstance(actions, np.ndarray):
                actions = actions.astype(np.float32)
            elif isinstance(actions, torch.Tensor):
                actions = actions.float()
            inputs["actions"] = actions

        return inputs


@dataclasses.dataclass(frozen=True)
class CALVINOutputs(transforms.DataTransformFn):
    """Outputs for the CALVIN policy."""

    def __call__(self, data: dict) -> dict:
        # π₀ uses 7D actions (6 joint velocities + 1 gripper)
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": actions}
"""CALVIN policy transforms for π₀ (Pi0) - LeRobot format."""