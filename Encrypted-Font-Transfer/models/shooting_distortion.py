import torch
import torch.nn as nn
import kornia

class ShootingDistortion(nn.Module):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        
    def forward(self, encoded_image):
        # Gaussian Blur
        kernel_size = (7, 7)
        sigrange_gauss = torch.tensor([1.0, 3.0])
        sigma = torch.rand(2) * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
        sigma = (sigma[0].item(), sigma[1].item())
        kernel = kornia.filters.get_gaussian_kernel2d(kernel_size, sigma)
        kernel = kernel.unsqueeze(0)
        encoded_image = kornia.filter2d(encoded_image, kernel)

        # Noise
        rnd_noise = 0.02
        random_noise = torch.randn_like(encoded_image) * rnd_noise
        encoded_image += random_noise
        encoded_image = torch.clip(encoded_image, 0, 1)
        
        # Contrast
        contrast_low = 0.5
        contrast_high = 1.5
        contrast_params = torch.tensor([contrast_low, contrast_high])
        contrast_scale = torch.rand(self.batch_size, 1, 1, 1, device=encoded_image.device) * (contrast_params[1] - contrast_params[0]) + contrast_params[0]
        encoded_image *= contrast_scale
        
        # Brightness
        rnd_bri = 0.3
        rnd_brightness = torch.rand(self.batch_size, 1, 1, 1, device=encoded_image.device) * 2 * rnd_bri - rnd_bri
        encoded_image += rnd_brightness
        
        # Hue
        rnd_hue = 0.1
        rnd_hue = torch.rand(encoded_image.size(), device=encoded_image.device) * 2 * rnd_hue - rnd_hue
        encoded_image += rnd_hue
        encoded_image = torch.clip(encoded_image, 0, 1)
        
        # Saturation
        rnd_sat = 1.0
        encoded_image_lum = encoded_image.mean(dim=1, keepdim=True)
        encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum
        
        # JPEG Compression
        
        return encoded_image
