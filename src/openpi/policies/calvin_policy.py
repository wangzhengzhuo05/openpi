# src/openpi/policies/calvin_policy.py
import dataclasses
import numpy as np
import einops
from openpi import transforms

@dataclasses.dataclass(frozen=True)
class CALVINInputs(transforms.DataTransformFn):
    """Inputs for CALVIN (统一顶层schema: images/state/actions/prompt)."""

    def __call__(self, data: dict) -> dict:
        # 1) 读取顶层字段（与 RepackTransform 输出一致）
        in_images = data["images"]          # dict: base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb
        state = np.asarray(data["state"], dtype=np.float32)[[6, 7, 8, 9, 10, 11, 12]]   # 1D
        actions = data.get("actions")       # [T, 7] or None
        prompt = data.get("prompt")

        # 2) 图片规范化：CHW→HWC，浮点→uint8
        def cvt(img):
            arr = np.asarray(img)
            if np.issubdtype(arr.dtype, np.floating):
                arr = (255 * arr).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = einops.rearrange(arr, "c h w -> h w c")
            return arr

        base = cvt(in_images["base_0_rgb"])
        left = cvt(in_images["left_wrist_0_rgb"])

        images = {
            "base_0_rgb": base,
            "left_wrist_0_rgb": left,
            # 右腕通道按你的要求 mask 掉；内容放零
            "right_wrist_0_rgb": np.zeros_like(base),
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,  # ← 关键：mask 掉
        }

        out = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }
        if actions is not None:
            out["actions"] = np.asarray(actions)
        if prompt is not None:
            out["prompt"] = prompt
        else:
            print("No prompt provided in CALVINInputs.")
        return out

@dataclasses.dataclass(frozen=True)
class CALVINOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 只取前7维（6关节+夹爪）
        return {"actions": np.asarray(data["actions"])[:, :7]}
