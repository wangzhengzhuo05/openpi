"""CALVIN policy transforms for PI0."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_calvin_example() -> dict:
    """Creates a random input example for the CALVIN policy."""
    return {
        "state": np.random.randn(15).astype(np.float32),  # 15D robot state (+ tactile if enabled)
        "images": {
            "rgb_static": np.random.randint(256, size=(3, 200, 200), dtype=np.uint8),
            "rgb_gripper": np.random.randint(256, size=(3, 84, 84), dtype=np.uint8),
        },
        "prompt": "move the light switch to turn on the yellow light",
    }


@dataclasses.dataclass(frozen=True)
class CALVINInputs(transforms.DataTransformFn):
    """Inputs for the CALVIN policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [15 + tactile_dims] - robot joint positions/velocities/gripper + flattened tactile
    - actions: [action_horizon, 7] - 6 joint velocities + 1 gripper command
    - prompt: language instruction string
    
    Note: Tactile data is stored as flattened components in the state vector:
    state = [robot_state (15D), tactile_rgb_flat, tactile_depth_flat]
    """

    # CALVIN uses Franka Panda robot
    # State: 15D robot + optional tactile (flattened)
    # Actions: 7D (6 joint velocities + 1 gripper command)
    
    # The expected camera names - now including depth
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("rgb_static", "rgb_gripper", "depth_static", "depth_gripper")
    
    # Whether to include tactile data (if False, will use only first 15 dims of state)
    use_tactile: bool = True

    def __call__(self, data: dict) -> dict:
        data = _decode_calvin(data, use_tactile=self.use_tactile)

        in_images = data["images"]
        
        # Check that required RGB cameras are present
        required_cameras = ("rgb_static", "rgb_gripper")
        missing = set(required_cameras) - set(in_images)
        if missing:
            raise ValueError(f"Missing required cameras: {missing}")

        # Primary camera is rgb_static
        base_image = in_images["rgb_static"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add wrist camera
        if "rgb_gripper" in in_images:
            images["left_wrist_0_rgb"] = in_images["rgb_gripper"]
            image_masks["left_wrist_0_rgb"] = np.True_
        else:
            images["left_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["left_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CALVINOutputs(transforms.DataTransformFn):
    """Outputs for the CALVIN policy."""

    def __call__(self, data: dict) -> dict:
        # CALVIN uses 7D actions (6 joint velocities + 1 gripper)
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": actions}


def _decode_calvin(data: dict, use_tactile: bool = True) -> dict:
    """Decode CALVIN data format.
    
    CALVIN state: [15 + tactile_dims] - robot + flattened tactile
    CALVIN actions: [action_horizon, 7] - joint velocities + gripper command
    
    Args:
        data: Input data dictionary
        use_tactile: If False, will use only first 15 dims of state (robot only)
    """
    
    # State may include flattened tactile data
    state = np.asarray(data["state"])
    
    # If not using tactile, truncate to first 15 dims (robot state only)
    if not use_tactile and len(state) > 15:
        state = state[:15]

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel]
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW format
            return einops.rearrange(img, "c h w -> h w c")
        return img

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data