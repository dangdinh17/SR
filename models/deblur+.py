import cv2
import numpy as np
import os
from PIL import Image
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ResidualBlock(nn.Module):
    def __init__(self, num_chanels):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_chanels, num_chanels, kernel_size=3, padding=1, stride=1),
#             nn.Dropout(0.5), 
            nn.InstanceNorm2d(num_chanels), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_chanels, num_chanels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(num_chanels)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class Downsample(nn.Module):
    def __init__(self, input_chanels, output_chanels):
        super(Downsample,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_chanels, output_chanels, kernel_size=2, stride=2),
            nn.InstanceNorm2d(output_chanels), 
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.block(x)
    
class Upsample(nn.Module):
    def __init__(self, input_chanels, output_chanels):
        super(Upsample, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_chanels, output_chanels, kernel_size=2, stride=2),
            nn.InstanceNorm2d(output_chanels), 
            nn.LeakyReLU(0.2),
        )

    
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, num_channels = 3):
        super(Generator, self).__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1, stride = 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.downsample = Downsample(64, 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride = 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        ) 
        self.resblock = self.residual = nn.Sequential(
#             *[ResidualBlock(128) for _ in range(4)],
#             nn.Conv2d(128, 256, kernel_size=3, padding=1, stride = 1),
#             nn.LeakyReLU(0.2),
            *[ResidualBlock(256) for _ in range(9)],
#             nn.Conv2d(256, 128, kernel_size=3, padding=1, stride = 1),
#             nn.LeakyReLU(0.2),
#             *[ResidualBlock(128) for _ in range(4)],
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride = 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
        ) 
        self.upsample = Upsample(128, 64)
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1, stride = 1),
            nn.Tanh(),
        )
        
        
    def forward(self, x):
        input = self.input_conv(x)
        input = self.downsample(input)
        input = self.conv1(input)
        input = self.resblock(input)
        input = self.conv2(input)
        input = self.upsample(input)
        input = self.output_conv(input)
        return x + input

device = torch.device('cpu')
gnet = Generator()
gnet.load_state_dict(torch.load('weight/best_gnet.pth', map_location = device))
gnet.to(device)

lr_image_path = 'real_test/test1_150x150_motionblured_3\images\l_light_09_spurious_copper_10_4_150_mt_3.jpg'
transform = transforms.ToTensor()

lr_image = Image.open(lr_image_path)
# hr_image = Image.open(hr_image_path)

lr_image = transform(lr_image).unsqueeze(0).to(device)  # Thêm batch dimension và chuyển sang CPU
# hr_image = transform(hr_image).unsqueeze(0).to(device)  # Thêm batch dimension và chuyển sang CPU

# Dự đoán
output = gnet(lr_image)

# # Tính toán PSNR
# psnr = calculate_psnr(output, hr_image)
# psnr_values.append(psnr)

# Chuyển đổi tensor đầu ra thành ảnh và lưu
output_image = output.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
output_image = transforms.ToPILImage()(output_image)  # Chuyển tensor thành ảnh PIL
output_image.save('output.jpg')  # Lưu ảnh