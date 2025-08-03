import os
import yaml
import numpy as np
import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import torch
import monai.transforms as tr
from monai.data import Dataset, CacheDataset, DataLoader, ThreadDataLoader
from monai.utils import first, set_determinism, ensure_tuple_rep

from utils import SliceWiseNormalizeIntensityd
from utils import read_config, visualize_sample
import warnings

warnings.filterwarnings('ignore')


class FetalDataLoader:
    """
    Data loader for fetal imaging segmentation tasks.
    Handles both 2D and 3D medical images with various transformations and augmentations.
    """

    def __init__(self, config, Train=True):
        """
        Initialize the data loader with configuration settings.

        Args:
            config: Configuration dictionary or object containing dataset parameters
            Train: Boolean flag to indicate training (True) or testing (False) mode
        """
        self.config = config
        self.Train = Train

    def train_transformations_2d(self):
        """
        Define transformations for 2D image training and validation.

        Returns:
            tuple: (train_transforms, val_transforms) composed transformation pipelines
        """
        # Basic transformations for training data
        train_trans = [
            tr.LoadImaged(keys=["image", "label"]),  # Load image and label from files
            tr.EnsureChannelFirstd(keys=["image", "label"]),  # Set channel dimension as first
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            # Resample to uniform spacing
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),  # Remove singleton dimensions
            tr.NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),  # Normalize intensities
            tr.Resized(keys=["image", "label"], spatial_size=(self.config["img_size"], self.config["img_size"])),
            # Resize to target dimensions
        ]

        # Basic transformations for validation data
        val_trans = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
            SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
            # Custom normalization
            tr.Resized(keys=["image", "label"], spatial_size=(self.config["img_size"], self.config["img_size"], -1)),
            # Keep z dimension as is
        ]

        # Add augmentations for training if enabled in config
        if self.config["augmentation"]:
            # Spatial augmentations (pick one randomly)
            spatial_aug = tr.OneOf([
                # tr.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                # tr.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.5),
                # tr.RandRotated(keys=["image", "label"], range_x=(0.2, 1.0), prob=0.6),
                tr.RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, prob=0.6),  # Random zoom
                tr.RandAffined(keys=["image", "label"], padding_mode="zeros",
                               rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6), shear_range=(0.5, 0.5, 0.5),
                               translate_range=(5, 5, 5), mode=("bilinear", "nearest"), prob=0.6),
                # Random affine transformation
            ])

            # Intensity augmentations (pick one randomly)
            intensity_aug = tr.OneOf([
                tr.RandGaussianNoised(keys=["image"], mean=0, std=0.1, prob=0.5),  # Add Gaussian noise
                tr.RandBiasFieldd(keys=["image"], degree=4, coeff_range=(0.05, 0.1), prob=0.2),  # Add bias field
                # tr.RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.4),
                # CustomSuperpixelMask(keys=["image", "label"], n_segments=[2, 3], compactness=0.09, random_range=(-1, 1), p=1)
            ])

            train_trans.append(spatial_aug)
            train_trans.append(intensity_aug)

        return tr.Compose(train_trans), tr.Compose(val_trans)

    def train_transformations_3d(self):
        """
        Define transformations for 3D volume training and validation.

        Returns:
            tuple: (train_transforms, val_transforms) composed transformation pipelines
        """
        # Basic transformations for training data
        train_trans = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(
                keys=["image", "label"],
                pixdim=(
                    self.config.voxel_spacing[0],
                    self.config.voxel_spacing[1],
                    self.config.voxel_spacing[2]
                ),
                mode=("bilinear", "nearest")
            ),
            # Resizing with padding/cropping is used instead of simple resize
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                    self.config.img_size[2]
                ),
                method="symmetric",
                mode="constant",
            ),
        ]

        # Add augmentations for training if enabled in config
        if self.config.augmentation:
            spatial_aug = tr.OneOf([
                tr.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.6),
                tr.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.6),
                tr.RandRotated(keys=["image", "label"], range_x=(0.2, 1.0), prob=0.6),
                tr.RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.3, prob=0.6),
                tr.RandAffined(keys=["image", "label"], padding_mode="zeros",
                               rotate_range=(np.pi / 4, np.pi / 4), shear_range=(0.5, 0.5),
                               translate_range=(30, 30), mode=("bilinear", "nearest"), prob=0.6),
            ])

            intensity_aug = tr.OneOf([
                tr.RandGaussianNoised(keys=["image"], mean=0, std=0.4, prob=0.5),
                tr.RandBiasFieldd(keys=["image"], degree=4, coeff_range=(0.05, 0.1), prob=0.6),
                tr.RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.4),
            ])

            train_trans.extend([spatial_aug, intensity_aug])

        # Add normalization after any augmentations
        train_trans.append(
            tr.NormalizeIntensityd(
                keys=["image"],
                subtrahend=0.0,
                divisor=None,
                nonzero=True
            )
        )

        # Validation transformations (without augmentations)
        val_trans = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(
                keys=["image", "label"],
                pixdim=(
                    self.config.voxel_spacing[0],
                    self.config.voxel_spacing[1],
                    self.config.voxel_spacing[2]
                ),
                mode=("bilinear", "nearest")
            ),
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                    self.config.img_size[2]
                ),
                method="symmetric",
                mode="constant",
            ),
            tr.NormalizeIntensityd(
                keys=["image"],
                subtrahend=0.0,
                divisor=None,
                nonzero=True
            ),
        ]

        return tr.Compose(train_trans), tr.Compose(val_trans)

    def test_transformations_2d(self):
        """
        Define transformations for 2D image testing.

        Returns:
            list: List of transformations for test data
        """
        test_transforms_list = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
            SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
            # tr.Resized(keys="image", spatial_size=(self.config["img_size"], self.config["img_size"], -1))
            # tr.ResizeWithPadOrCropd(keys="image", spatial_size=(self.config["img_size"], self.config["img_size"], -1)),
        ]
        return test_transforms_list

    def test_transformations_3d(self):
        """
        Define transformations for 3D volume testing.

        Returns:
            list: List of transformations for test data
        """
        test_transforms_list = [
            tr.LoadImaged(
                keys=["image", "label"]
            ),
            tr.EnsureChannelFirstd(
                keys=["image", "label"]
            ),
            tr.Spacingd(
                keys=["image", "label"],
                pixdim=(
                    self.config.voxel_spacing[0],
                    self.config.voxel_spacing[1],
                    self.config.voxel_spacing[2]
                ),
                mode=("bilinear", "nearest")
            ),
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                    self.config.img_size[2]
                ),
                method="symmetric",
                mode="constant",
            ),
            tr.NormalizeIntensityd(
                keys=["image"],
                subtrahend=0.0,
                divisor=None,
                nonzero=True
            ),
        ]
        return test_transforms_list

    def load_data(self):
        """
        Load datasets and create data loaders based on configuration.

        Returns:
            If Train=True: tuple of (train_dataloader, val_dataloader)
            If Train=False: tuple of (test_dataloader, test_files, test_transforms_list)
        """
        if self.Train:
            # Select appropriate transformations based on dimensionality
            if self.config.spatial_dims == "2d":
                train_transforms, val_transforms = self.train_transformations_2d()
            elif self.config.spatial_dims == "3d":
                train_transforms, val_transforms = self.train_transformations_3d()

            # Load training data based on configuration
            if self.config.train_data_type == "path":
                # Load from directory structure
                train_images = sorted(glob(os.path.join(self.config.train_data_paths, 'images', "img_*.nii.gz")))
                train_labels = sorted(glob(os.path.join(self.config.train_data_paths, 'masks', "mask_*.nii.gz")))
            elif self.config.train_data_type == "file":
                # Load from CSV file listing
                train_images = []
                train_labels = []
                for path in self.config.train_data_paths:
                    data = pd.read_csv(path)
                    train_images += data.iloc[:, 0].tolist()
                    train_labels += data.iloc[:, 1].tolist()

            # Create dictionary of file pairs
            train_files = [{"image": image_name, "label": label_name} for
                           image_name, label_name in zip(train_images, train_labels)]

            # Load validation data based on configuration
            if self.config.validation_data_type == "path":
                # Load from directory structure
                val_images = sorted(glob(os.path.join(self.config.val_data_paths, 'images', "img_*.nii.gz")))
                val_labels = sorted(glob(os.path.join(self.config.val_data_paths, 'masks', "mask_*.nii.gz")))
            elif self.config.validation_data_type == "file":
                # Load from CSV file listing
                val_images = []
                val_labels = []
                for path in self.config.validation_data_paths:
                    data = pd.read_csv(path)
                    val_images += data.iloc[:, 0].tolist()
                    val_labels += data.iloc[:, 1].tolist()

            # Create dictionary of file pairs
            val_files = [{"image": image_name, "label": label_name} for
                         image_name, label_name in zip(val_images, val_labels)]

            # Create data loaders with appropriate settings
            if self.config.fast_training:
                # Use CacheDataset for better performance
                train_dataset = CacheDataset(data=train_files,
                                             transform=train_transforms,
                                             cache_rate=1.0,
                                             num_workers=8,
                                             copy_cache=False)

                train_dataloader = ThreadDataLoader(train_dataset,
                                                    num_workers=0,
                                                    batch_size=self.config.batch_size,
                                                    shuffle=True)

                val_dataset = CacheDataset(data=val_files,
                                           transform=val_transforms,
                                           cache_rate=1.0,
                                           num_workers=5,
                                           copy_cache=False)

                val_dataloader = ThreadDataLoader(val_dataset,
                                                  num_workers=0,
                                                  batch_size=1,
                                                  shuffle=False)
            else:
                # Use standard Dataset
                train_dataset = Dataset(data=train_files,
                                        transform=train_transforms)

                train_dataloader = DataLoader(train_dataset,
                                              batch_size=self.config.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=False)

                val_dataset = Dataset(data=val_files,
                                      transform=val_transforms)

                val_dataloader = DataLoader(val_dataset,
                                            batch_size=1,
                                            num_workers=4,
                                            shuffle=False)

            return train_dataloader, val_dataloader

        else:
            # Testing mode - load test data
            if self.config.test_data_type == "path":
                # Recursive search through directory structure for test data
                test_images = sorted(glob(os.path.join(self.config.test_data_paths, 'data', '**/*.nii.gz'),
                                          recursive=True))
                test_labels = sorted(glob(os.path.join(self.config.test_data_paths, 'manual-masks', '**/*.nii.gz'),
                                          recursive=True))
            elif self.config.test_data_type == "file":
                # Load from CSV file listing
                test_images = []
                test_labels = []
                for path in self.config.test_data_paths:
                    data = pd.read_csv(path)
                    test_images += data.iloc[:, 0].tolist()
                    test_labels += data.iloc[:, 1].tolist()

            # Create dictionary of file pairs
            test_files = [{"image": image_name, "label": label_name} for
                          image_name, label_name in zip(test_images, test_labels)]

            # Select appropriate transformations based on dimensionality
            if self.config.spatial_dims == "2d":
                test_transforms_list = self.test_transformations_2d()
            else:
                test_transforms_list = self.test_transformations_3d()

            # Create test dataset and loader
            test_dataset = Dataset(data=test_files, transform=tr.Compose(test_transforms_list))
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0)

            return test_dataloader, test_files, test_transforms_list


if __name__ == '__main__':
    # Example usage and validation of the data loader
    import SimpleITK as sitk

    # Set up device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    configs = read_config('./config.yml', mode="train")

    # Override some config settings for testing
    configs.fast_training = False
    configs.augmentation = False

    # Create data loader
    print("Initializing data loader...")
    fetal_data = FetalDataLoader(configs)
    train_data_loader, val_data_loader = fetal_data.load_data()

    # Get a sample from the data loader for visualization
    print("Loading a sample batch for visualization...")
    i = 0  # Index of the batch sample to visualize
    check_data = first(train_data_loader)
    image, label = (check_data["image"][i][0], check_data["label"][i][0])
    print(f"Sample image shape: {image.shape}, label shape: {label.shape}")

    # Save sample as NIfTI file for external viewing
    output_path = 'test.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(image), output_path)
    print(f"Sample image saved to {output_path}")

    # Visualize the sample data
    print("\nVisualizing sample data...")
    visualize_sample(image, label, display=True)
