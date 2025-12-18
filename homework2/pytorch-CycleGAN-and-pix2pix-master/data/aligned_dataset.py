import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
import numpy as np
import torch

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert self.opt.load_size >= self.opt.crop_size  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        # 1. 读取并分割图像
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size
        w2 = int(w / 2)
        # A_pil 永远是真实照片, B_pil 永远是语义图
        A_pil = AB.crop((0, 0, w2, h))
        B_pil = AB.crop((w2, 0, w, h))

        # 2. 获取适用于所有图像的变换参数
        transform_params = get_params(self.opt, A_pil.size)

        # 3. 对 A (照片) 和 B (语义图) 进行基础变换
        #    注意 grayscale 的判断逻辑
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1))
        tensor_A = A_transform(A_pil)
        tensor_B = B_transform(B_pil)

        # 4. 如果开启边缘引导，修改 tensor_B
        if self.opt.use_edge_map:
            # a. 从真实照片 A 生成边缘图
            A_np = np.array(A_pil)
            A_gray = cv2.cvtColor(A_np, cv2.COLOR_RGB2GRAY)
            edges_np = cv2.Canny(A_gray, 100, 200)
            edge_pil = Image.fromarray(edges_np)

            # b. 对边缘图应用同样的变换 (必须是 grayscale)
            edge_transform = get_transform(self.opt, transform_params, grayscale=True)
            edge_tensor = edge_transform(edge_pil)

            # c. 将 3 通道的 tensor_B 和 1 通道的 edge_tensor 拼接
            tensor_B = torch.cat([tensor_B, edge_tensor], 0)

        # 5. 固定返回：'A' 是照片, 'B' 是语义图(可能是4通道)
        #    模型层会根据 --direction BtoA 自己选择 'B' 作为输入
        return {"A": tensor_A, "B": tensor_B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
