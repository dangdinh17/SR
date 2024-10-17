import torch
import torch.nn as nn
import math
import numpy as np
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

class ResidualCatBlock(nn.Module):
    def __init__(self, num_channels=64):
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


class EDRN(nn.Module):
    def __init__(self, num_channels = 3, use_canny=False, use_sobel=False):
        super(EDRN, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.use_canny = use_canny
        self.use_sobel = use_sobel
        
        self.canny_filter = CannyFilterOpenCV() if use_canny else None
        self.sobel_filter = SobelFilterOpenCV() if use_sobel else None
        
        if use_canny:
            additional_channels = 1
        if use_sobel:
            additional_channels = 3
        self.input_conv = nn.Conv2d(num_channels+additional_channels, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(ResidualCatBlock, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv_input(out)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        
        return out

