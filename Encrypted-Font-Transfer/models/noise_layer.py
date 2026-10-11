from logging.handlers import RotatingFileHandler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from params import DEVICE
from PIL import Image
from models.jpeg_compression import JpegCompression
from models.shooting_distortion import ShootingDistortion

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image_height, image_width, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Identity(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input_image):
        return input_image


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range


    def forward(self, input_image):
        _, _, H, W = input_image.shape
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(H, W, self.height_ratio_range, self.width_ratio_range)

        input_image[:, :, h_start: h_end, w_start: w_end] = 0

        return input_image


class AddSaltPepper(nn.Module):
    def __init__(self, density, device):
        super(AddSaltPepper, self).__init__()
        self.density = density
        self.device = device

    def forward(self, input_image):
        _, c, h, w = input_image.shape
        Nd = self.density
        mask = torch.rand(h, w).repeat(1, c, 1, 1).to(self.device)
        salt = torch.ones(1, c, h, w).to(self.device)
        pepper = torch.zeros(1, c, h, w).to(self.device)
        input_image = torch.where(mask < Nd/2, salt, input_image)
        input_image = torch.where(mask > 1 - Nd/2, pepper, input_image)
        return input_image


class AddGaussian(nn.Module):
    def __init__(self, device):
        super(AddGaussian, self).__init__()
        self.device = device

    def forward(self, input_image):
        _, c, h, w = input_image.shape
        N = 0.5 * torch.randn(h, w).repeat(1, c, 1, 1).to(self.device)
        input_image = N + input_image
        input_image[input_image > 1] = 1
        return input_image


class RandomRotate(nn.Module):
    """
    Randomly rotates the image
    """
    def __init__(self, angle_range):
        super(RandomRotate, self).__init__()
        self.rotate = transforms.RandomRotation(degrees=angle_range)

    def forward(self, input_image):
        return self.rotate(input_image)
    

class GaussianBlur(nn.Module):
    """Apply Gaussian blur with a configurable sigma range."""
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        super(GaussianBlur, self).__init__()
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, input_image):
        return self.blur(input_image)


class MotionBlur(nn.Module):
    """Apply a directional kernel independently to every image channel."""
    def __init__(self, kernel_size=5, angle=30, device=DEVICE):
        super(MotionBlur, self).__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.device = device
        self.register_buffer('kernel', self._build_kernel(kernel_size, angle))

    def _build_kernel(self, k, angle):
        import numpy as np
        kernel = np.zeros((k, k), dtype=np.float32)
        center = k // 2
        theta = np.deg2rad(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        for i in range(k):
            for j in range(k):
                if abs((i - center) * direction[0] + (j - center) * direction[1]) < 0.5:
                    kernel[i, j] = 1.0
        s = kernel.sum()
        if s > 0:
            kernel /= s
        kernel = torch.from_numpy(kernel).float().to(self.device)
        return kernel

    def forward(self, input_image):
        b, c, h, w = input_image.shape
        k = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(input_image, k, padding=self.kernel_size // 2, groups=c)


class Perspective(nn.Module):
    """Apply a random perspective transform."""
    def __init__(self, distortion_scale=0.3, p=1.0):
        super(Perspective, self).__init__()
        self.warp = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p)

    def forward(self, input_image):
        return self.warp(input_image)


class ResolutionChange(nn.Module):
    """Downsample and restore an image with bilinear interpolation."""
    def __init__(self, scale_range=(0.5, 0.8)):
        super(ResolutionChange, self).__init__()
        self.scale_range = scale_range

    def forward(self, input_image):
        b, c, h, w = input_image.shape
        scale = random_float(self.scale_range[0], self.scale_range[1])
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        small = F.interpolate(input_image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        back = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
        return back


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self):
        super(Noiser, self).__init__()
        self.device = DEVICE
        
        self.noise_layers = [
            Identity(),
            Crop((0.2, 0.4), (0.2, 0.4)),
            AddSaltPepper(0.1, self.device),
            AddGaussian(self.device),
            GaussianBlur(kernel_size=3, sigma=(0.3, 1.2)),
            MotionBlur(kernel_size=5, angle=25, device=self.device),
            JpegCompression(self.device, mode='DctSimulation'),
            RandomRotate((10, 45)),
            Perspective(distortion_scale=0.3, p=1.0),
            ResolutionChange(scale_range=(0.5, 0.8)),
            ShootingDistortion(8),
        ]

    def forward(self, input_image, id=None):
        if id is None:
            selected_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        else:
            selected_noise_layer = self.noise_layers[id]
        return selected_noise_layer(input_image)
