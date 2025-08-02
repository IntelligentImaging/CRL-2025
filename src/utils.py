import math

import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import slic

import torch
from monai.transforms import MapTransform, Transform

import warnings

warnings.filterwarnings('ignore')


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def read_config(file_path, mode='train'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    general_config = {k: v for k, v in config.items() if (k != 'train_mode' and k != 'test_mode')}

    if mode == 'train':
        train_config = config['train_mode']
        combined_config = {**general_config, **train_config}
    else:
        train_config = config['test_mode']
        combined_config = {**general_config, **train_config}

    return Config(combined_config)

config = read_config('./config.yml', mode='test')


def generate_random_number(low, high):
    random_number = torch.rand(1) * (high - low) + low
    return random_number.item()


class SliceWiseNormalizeIntensityd(MapTransform):
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                if self.nonzero:
                    mask = slice_ > 0
                    if np.any(mask):
                        if self.subtrahend is None:
                            slice_[mask] = slice_[mask] - slice_[mask].mean()
                        else:
                            slice_[mask] = slice_[mask] - self.subtrahend

                        if self.divisor is None:
                            slice_[mask] /= slice_[mask].std()
                        else:
                            slice_[mask] /= self.divisor

                else:
                    if self.subtrahend is None:
                        slice_ = slice_ - slice_.mean()
                    else:
                        slice_ = slice_ - self.subtrahend

                    if self.divisor is None:
                        slice_ /= slice_.std()
                    else:
                        slice_ /= self.divisor

                image[..., i] = slice_
            d[key] = image
        return d


class CustomSuperpixelMask(Transform):
    def __init__(self, keys, n_segments, compactness, random_range=(-1, 1), p=0.5):
        self.img_key = keys[0]
        self.mask_key = keys[1]
        self.n_segments = np.random.randint(n_segments[0], n_segments[1])
        self.compactness = compactness
        self.random_range = random_range
        self.p = p

    def __call__(self, data):
        if torch.rand(1) > self.p:
            return data

        img = data[self.img_key][0]
        mask = data[self.mask_key][0].bool()

        if mask.sum() > 500:
            segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, mask=mask)
            segment_vals = np.unique(segments[segments != 0])

            selected_segment = np.random.choice(segment_vals)
            segment_mask = torch.from_numpy(segments == selected_segment)

            intensity_option = np.random.choice(['brighten', 'blur', 'noise', 'darken'])
            if intensity_option == 'brighten':
                factor_map = np.random.uniform(1.05, 1.2, size=data[self.img_key][:, mask & segment_mask].shape)
                factor_map = gaussian_filter(factor_map, sigma=0.4)
                data[self.img_key][:, mask & segment_mask] *= torch.from_numpy(factor_map).to(data[self.img_key].device)

            elif intensity_option == 'blur':
                blurred_segment = gaussian_filter(img.cpu().numpy(), sigma=generate_random_number(0.9, 1.0))
                data[self.img_key][:, mask & segment_mask] = \
                    torch.from_numpy(blurred_segment).to(data[self.img_key].device)[mask & segment_mask]

            elif intensity_option == 'noise':
                random_noise = torch.normal(mean=0, std=generate_random_number(0.2, 0.4),
                                            size=((mask & segment_mask).sum().item(),)).type(data[self.img_key].dtype)
                random_noise = gaussian_filter(random_noise, sigma=1.0)

                data[self.img_key][:, mask & segment_mask] += torch.from_numpy(random_noise)

            elif intensity_option == 'darken':
                factor_map = np.random.uniform(0.7, 0.8, size=data[self.img_key][:, mask & segment_mask].shape)
                factor_map = gaussian_filter(factor_map, sigma=0.4)
                data[self.img_key][:, mask & segment_mask] *= torch.from_numpy(factor_map).to(data[self.img_key].device)

        return data


def visualize_sample(images, masks, gt=None,
                volume_dice=None, mean_slice_dice=None, slice_dice=None,
                save_name=None, display=None):
    slice_num = images.shape[-1]
    # Calculate the number of columns and rows based on the number of images
    n_cols = int(math.ceil(math.sqrt(slice_num)))
    n_rows = int(math.ceil(slice_num / n_cols))


    cmap_mask = matplotlib.colors.ListedColormap(['none', 'red'])
    cmap_gt = matplotlib.colors.ListedColormap(['none', 'blue'])

    # Create a grid of subplots with the calculated number of columns and rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    for i in range(slice_num):
        # Calculate the row and column indices for the current subplot
        row = i // n_cols
        col = i % n_cols

        # Plot the image with the mask overlay
        axs[row, col].imshow(images[:, :, i], cmap='gray')
        axs[row, col].imshow(masks[:, :, i], alpha=0.6, cmap=cmap_mask)

        if not (gt is None):
            axs[row, col].imshow(gt[:, :, i], alpha=0.3, cmap=cmap_gt)

        if not (slice_dice is None):
            axs[row, col].set_title(f"dice= {slice_dice[i]:.2f}")

        axs[row, col].axis('off')

    # Remove any unused subplots
    for i in range(len(images), n_rows * n_cols):
        axs.flatten()[i].set_visible(False)

    if volume_dice and mean_slice_dice:
        fig.suptitle(
            f"red: predicted, blue: manual, volume_dice= {volume_dice:.2f}, mean_slice_dice= {mean_slice_dice:.2f}")

    elif volume_dice:
        fig.suptitle(
            f"red: predicted, blue: manual, volume_dice= {volume_dice:.2f}")

    if save_name:
        plt.savefig(save_name)

    if display:
        plt.show()
