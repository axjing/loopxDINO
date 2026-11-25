"""
Data pipeline for Pascal VOC 2012 Semantic Segmentation.
Supports official VOCSegmentation and synthetic fallback for fast smoke-testing.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
NUM_VOC_CLASSES = 21
IGNORE_LABEL = 255


class VOCSegmentationDataset(Dataset):
    """
    Pascal VOC 2012 Segmentation Dataset with aligned spatial augmentations.
    """

    def __init__(
        self,
        root: str = "data/pascal_voc",
        image_set: str = "train",
        img_size: int = 512,
        download: bool = False,
        num_queries: int = 21,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.root = root
        self.image_set = image_set
        self.img_size = img_size
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        try:
            from torchvision.datasets import VOCSegmentation

            self.dataset = VOCSegmentation(
                root=root,
                year="2012",
                image_set=image_set,
                download=download,
            )
            self.is_synthetic = False
        except Exception as e:  # noqa: BLE001
            # Fallback if download is False and dataset directory does not exist
            print(
                f"[VOCDataLoader] Warning: Could not load real VOC dataset ({e}). Falling back to SyntheticVOCDataset."
            )
            self.dataset = None
            self.is_synthetic = True
            self.synthetic_ds = SyntheticVOCDataset(
                num_samples=64 if image_set == "train" else 16,
                img_size=img_size,
                num_queries=num_queries,
                embed_dim=embed_dim,
            )

    def __len__(self) -> int:
        if self.is_synthetic:
            return len(self.synthetic_ds)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_synthetic:
            return self.synthetic_ds[idx]

        image, target = self.dataset[idx]

        # Aligned spatial augmentations for training
        if self.image_set == "train":
            # 1. Random horizontal flip (p=0.5)
            if random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)

            # 2. Mild random scaling (0.9 to 1.1) with crop / pad
            scale = random.uniform(0.9, 1.1)
            new_size = int(self.img_size * scale)
            image = TF.resize(
                image, [new_size, new_size], interpolation=InterpolationMode.BILINEAR
            )
            target = TF.resize(
                target, [new_size, new_size], interpolation=InterpolationMode.NEAREST
            )

            if new_size > self.img_size:
                top = random.randint(0, new_size - self.img_size)
                left = random.randint(0, new_size - self.img_size)
                image = TF.crop(image, top, left, self.img_size, self.img_size)
                target = TF.crop(target, top, left, self.img_size, self.img_size)
            elif new_size < self.img_size:
                pad_h = self.img_size - new_size
                pad_w = self.img_size - new_size
                p_t = pad_h // 2
                p_b = pad_h - p_t
                p_l = pad_w // 2
                p_r = pad_w - p_l
                image = TF.pad(image, [p_l, p_t, p_r, p_b], fill=0)
                target = TF.pad(target, [p_l, p_t, p_r, p_b], fill=IGNORE_LABEL)
        else:
            image = TF.resize(
                image,
                [self.img_size, self.img_size],
                interpolation=InterpolationMode.BILINEAR,
            )
            target = TF.resize(
                target,
                [self.img_size, self.img_size],
                interpolation=InterpolationMode.NEAREST,
            )

        # Convert image to tensor and normalize with ImageNet stats
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Target mask to torch.long tensor
        target_tensor = torch.from_numpy(np.array(target, dtype=np.int64))
        if target_tensor.dim() == 3:
            target_tensor = target_tensor.squeeze(0)

        # Initial prompt/query tokens for reasoning decoder (num_queries, embed_dim)
        queries = torch.randn(self.num_queries, self.embed_dim, dtype=torch.float32)

        return img_tensor, queries, target_tensor


class SyntheticVOCDataset(Dataset):
    """
    High-fidelity synthetic Pascal VOC dataset for offline testing and verification.
    Generates synthetic multi-class segmentation masks with boundary void pixels.
    """

    def __init__(
        self,
        num_samples: int = 64,
        img_size: int = 512,
        num_queries: int = 21,
        embed_dim: int = 768,
        num_classes: int = 21,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Synthetic RGB image normalized
        image = torch.randn(3, self.img_size, self.img_size, dtype=torch.float32)

        # Synthetic segmentation mask with background (0) and 2-4 foreground objects
        mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
        num_objects = (idx % 3) + 2
        for obj_idx in range(num_objects):
            cls_id = ((idx * 3 + obj_idx) % (self.num_classes - 1)) + 1
            # Random box region
            cx = (self.img_size // (num_objects + 1)) * (obj_idx + 1)
            cy = self.img_size // 2
            r = self.img_size // 6
            y_min, y_max = max(0, cy - r), min(self.img_size, cy + r)
            x_min, x_max = max(0, cx - r), min(self.img_size, cx + r)
            mask[y_min:y_max, x_min:x_max] = cls_id
            # Add boundary ignore label around object edges
            mask[y_min : y_min + 2, x_min:x_max] = IGNORE_LABEL
            mask[y_max - 2 : y_max, x_min:x_max] = IGNORE_LABEL
            mask[y_min:y_max, x_min : x_min + 2] = IGNORE_LABEL
            mask[y_min:y_max, x_max - 2 : x_max] = IGNORE_LABEL

        # Initial prompt/query tokens
        queries = torch.randn(self.num_queries, self.embed_dim, dtype=torch.float32)
        return image, queries, mask


def get_voc_dataloaders(
    root: str = "data/pascal_voc",
    batch_size: int = 4,
    img_size: int = 512,
    download: bool = False,
    force_synthetic: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates train and val DataLoaders for Pascal VOC 2012.
    """
    if force_synthetic:
        train_ds = SyntheticVOCDataset(num_samples=64, img_size=img_size)
        val_ds = SyntheticVOCDataset(num_samples=16, img_size=img_size)
    else:
        train_ds = VOCSegmentationDataset(
            root=root, image_set="train", img_size=img_size, download=download
        )
        val_ds = VOCSegmentationDataset(
            root=root, image_set="val", img_size=img_size, download=download
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader
