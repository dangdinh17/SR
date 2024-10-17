import cv2
import numpy as np
import os
from PIL import Image
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class EDSR_modify(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4):
        super(EDSR_modify, self).__init__()
        self.scale_factor = scale_factor
        self.input_conv = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        x = self.input_conv(x)
        res = self.residual_layers(x)
        x = self.upsample(res)
        return x

class EDSR_orig(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4):
        super(EDSR_orig, self).__init__()
        self.scale_factor = scale_factor
        self.input_conv = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        x = self.input_conv(x)
        res = self.residual_layers(x)
        out = res + x
        out = self.upsample(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return x + self.block(x)

# Định nghĩa dataset
class ImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_image = cv2.imread(os.path.join(self.lr_dir, self.lr_files[idx]))
        hr_image = cv2.imread(os.path.join(self.hr_dir, self.hr_files[idx]))
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
        # Thay đổi kích thước ảnh tại đây nếu cần thiết
        # lr_image = cv2.resize(lr_image, (new_width, new_height))
        # hr_image = cv2.resize(hr_image, (new_width * scale_factor, new_height * scale_factor))
        
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1).float() / 255.0
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 255.0
        return lr_image, hr_image


class New_SR(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4):
        super(New_SR, self).__init__()
        self.scale_factor = scale_factor
        self.input_conv = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.residual_layers = nn.Sequential(
            *[ResidualCatBlock(64) for _ in range(16)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        x = self.input_conv(x)
        res = self.residual_layers(x)
        out = res + x
        out = self.upsample(out)
        return out



class ResidualCatBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualCatBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d(num_channels*2, num_channels, kernel_size = 1, stride = 1)
        
    def forward(self, x):
        out = self.block(x)
        out = torch.cat((x, out), 1)
        out = self.conv(out)
        return x + out

# Định nghĩa dataset

class CannyFilterOpenCV(nn.Module):
    def __init__(self, low_threshold=100, high_threshold=200):
        super(CannyFilterOpenCV, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, x):
        x_np = x.cpu().detach().numpy()
        canny_edges_batch = []
        for img in x_np:
            img_np = img.transpose(1, 2, 0)
            img_np = np.uint8(img_np * 255)
            canny_edges = cv2.Canny(img_np, self.low_threshold, self.high_threshold) # type: ignore
            canny_edges = canny_edges / 255.0
            canny_edges_batch.append(canny_edges[np.newaxis, ...])
        canny_edges_tensor = torch.from_numpy(np.array(canny_edges_batch)).float().to(x.device)
        return canny_edges_tensor
    
    def get_output_channels(self):
        return 1  # Canny filter luôn trả về 1 kênh
class SobelFilterOpenCV(nn.Module):
    def __init__(self):
        super(SobelFilterOpenCV, self).__init__()

    def forward(self, x):
        x_np = x.cpu().detach().numpy()
        sobel_edges_batch = []
        for img in x_np:
            img_np = img.transpose(1, 2, 0)
            img_np = np.uint8(img_np * 255)
            sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3) # type: ignore
            sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3) # type: ignore
            sobel_edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            sobel_edges = cv2.normalize(sobel_edges, None, 0, 1, cv2.NORM_MINMAX) # type: ignore
            sobel_edges_batch.append(sobel_edges.transpose(2, 0, 1))
        sobel_edges_tensor = torch.from_numpy(np.array(sobel_edges_batch)).float().to(x.device)
        return sobel_edges_tensor

    def get_output_channels(self, input_channels):
        return input_channels  # Sobel filter giữ nguyên số kênh đầu vào
class EDRN(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4, use_canny=False, use_sobel=False):
        super(EDRN, self).__init__()
        self.scale_factor = scale_factor
        self.use_canny = use_canny
        self.use_sobel = use_sobel
        
        self.canny_filter = CannyFilterOpenCV() if use_canny else None
        self.sobel_filter = SobelFilterOpenCV() if use_sobel else None
        
        if use_canny:
            additional_channels = 1
        if use_sobel:
            additional_channels = 3
        self.input_conv = nn.Conv2d(num_channels+additional_channels, 64, kernel_size=3, stride=1, padding=1)
        
        self.residual_layers = nn.Sequential(
            *[ResidualCatBlock(64) for _ in range(16)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        edge_maps = []
        if self.use_canny:
            edge_maps.append(self.canny_filter(x)) # type: ignore
        if self.use_sobel:
            edge_maps.append(self.sobel_filter(x)) # type: ignore
        
        if edge_maps:
            x_with_edges = torch.cat([x] + edge_maps, dim=1)
        else:
            x_with_edges = x
        
        x = self.input_conv(x_with_edges)
        res = self.residual_layers(x)
        out = res + x
        out = self.upsample(out)
        return out
