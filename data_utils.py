"""
Data preprocessing utilities for implicit neural representation training.
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torchaudio
from PIL import Image


def get_img_tensor(fname, sidelength):
    """
    Load and preprocess an image.
    Inspired by Sitzmann et al.
    
    Args:
        fname: Path to image file
        sidelength: Target size for the image
    
    Returns:
        Preprocessed image tensor
    """
    img = Image.open(fname)
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


def get_mgrid(sidelen, dim=2):
    """
    Generate a meshgrid for coordinates.
    Inspired by Sitzmann et al.
    
    Args:
        sidelen: Resolution of the grid
        dim: Dimensionality of the grid (default: 2)
    
    Returns:
        Flattened meshgrid tensor
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class PatternFitting(Dataset):
    """
    Dataset for fitting patterns (images).
    Inspired by Sitzmann et al.
    """
    
    def __init__(self, fname, sidelength, selected_channel=None):
        super().__init__()
        img = get_img_tensor(fname, sidelength)
        self.pixels = img.permute(1, 2, 0)
        if selected_channel is not None:
            self.pixels = self.pixels[:, :, selected_channel]
            self.pixels = torch.reshape(self.pixels, (1, sidelength**2, 1))
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels


class AudioFitting(Dataset):
    """Dataset for fitting audio signals."""
    
    def __init__(self, aud_file, selected_channel=None):
        self.aud_tensor, self.sample_rate = torchaudio.load(aud_file, format='mp3')
        self.channels, self.len = self.aud_tensor.shape
        self.aud_tensor = self.aud_tensor.reshape(1, self.len, self.channels)
        if selected_channel is not None:
            self.aud_tensor = self.aud_tensor[:, :, selected_channel].unsqueeze(2)
        self.coords = torch.linspace(-1, 1, steps=self.len)

    def get_time(self):
        return self.len

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.aud_tensor
