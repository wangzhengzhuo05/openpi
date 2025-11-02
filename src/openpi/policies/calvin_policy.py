"""CALVIN policy transforms for PI0."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_calvin_example() -> dict:
    """Creates a random input example for the CALVIN policy."""
    return {
        "state": np.random.randn(15).astype(np.float32),  # 15D robot state
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
    - state: [15] - robot joint positions and gripper state
    - actions: [action_horizon, 7] - 6 joint velocities + 1 gripper action
    - prompt: language instruction string
    """

    # CALVIN uses Franka Panda robot
    # State: 15D (joint positions, velocities, gripper state)
    # Actions: 7D (6 joint velocities + 1 gripper command)
    
    # The expected camera names. All input cameras must be in this set.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("rgb_static", "rgb_gripper")

    def __call__(self, data: dict) -> dict:
        data = _decode_calvin(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

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


def _decode_calvin(data: dict) -> dict:
    """Decode CALVIN data format.
    
    CALVIN state: [15] - robot joint positions, velocities, gripper
    CALVIN actions: [action_horizon, 7] - joint velocities + gripper command
    """
    
    # State is already in correct format (15D)
    state = np.asarray(data["state"])

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