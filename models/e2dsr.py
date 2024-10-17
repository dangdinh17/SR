import torch.nn as nn
import numpy as np
import cv2
import torch
import torchsummary
##########################################MainEDSRModel##########################################

class E2DSR(nn.Module):
    def __init__(self, num_channels=3, scale_factor=4,edge_option = 'sobel'):
        
        super(E2DSR, self).__init__()
        self.edgeoption = edge_option
        self.edgechannel = 0
        self.scale_factor = scale_factor

        if self.edgeoption =='sobel':
            self.edge_filter = SobelFilterOpenCV()
            self.edgechannel = self.edge_filter.get_output_channels()
        if self.edgeoption =='canny':
            self.edge_filter = CannyFilterOpenCV()
            self.edgechannel = self.edge_filter.get_output_channels()
        self.input_conv = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        self.edgelayer = HQNet(self.edgechannel)
        self.out = nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        
    
    def forward(self, x):
        #y = []
        y = self.edge_filter(x)
        # print(y.shape)
        x = self.input_conv(x)
        res = self.residual_layers(x)
        res1 = self.edgelayer(y)
        res = res + x
        res = self.upsample(res)
        out = res + res1 
        out = self.out(out)
        return out

##########################################ResidualBlock##########################################

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

##########################################EdgeFeatureEnhancementBlock##########################################
class HQNet(nn.Module):
    def __init__(self, num_channels):
        
        super(HQNet, self).__init__()
        self.path1 = nn.Sequential(
            nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None, recompute_scale_factor=None),
            nn.BatchNorm2d(num_channels),
            nn.SELU(),
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=None, scale_factor=4, mode='bicubic', align_corners=None, recompute_scale_factor=None)
        )
        self.out = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.SELU()
        )
        self.fuse = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        a = self.path1(x)
        b = self.path2(x)
        out = a * b
        out = self.out(out)
        out = self.fuse(out + b)
        return out 
    
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

    def get_output_channels(self):
        return 3
    
# model = E2DSR(edge_option='sobel').cpu()
# torchsummary.summary(model, (3, 150,150))