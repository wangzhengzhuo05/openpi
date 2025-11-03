import numpy as np
from openpi.transforms import Transform

class AddFakeRightWrist(Transform):
    """
    为 CALVIN 数据集补一个假的 right_wrist_0_rgb 通道（全 0 图像）。
    如果 batch 中已经有该通道，则跳过。
    """

    def __call__(self, batch):
        if "images" in batch and "right_wrist_0_rgb" not in batch["images"]:
            # 参考 base_0_rgb 的大小
            h, w, c = batch["images"]["base_0_rgb"].shape[-3:]
            batch["images"]["right_wrist_0_rgb"] = np.zeros((h, w, c), dtype=np.uint8)
        return batch
