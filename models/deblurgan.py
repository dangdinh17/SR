import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    def __init__(self, num_chanels):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_chanels, num_chanels, kernel_size=3, padding=1, stride=1),
            nn.Dropout(0.5), 
            nn.InstanceNorm2d(num_chanels), 
            nn.ReLU(),
            nn.Conv2d(num_chanels, num_chanels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(num_chanels)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class ConvBlock(nn.Module):
    def __init__(self, input_chanels, output_chanels):
        super(ConvBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_chanels, output_chanels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(output_chanels), 
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.block(x)
    
class TransposeBlock(nn.Module):
    def __init__(self, input_chanels, output_chanels):
        super(TransposeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_chanels, output_chanels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(output_chanels), 
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, num_channels = 3):
        super(Generator, self).__init__()
        
        self.input_conv = ConvBlock(3, 64)
        self.conv = nn.Sequential(
            *[ConvBlock(64*2**index, 64*2**(index+1)) for index in range(2)]
        )
        self.residual = nn.Sequential(
            *[ResidualBlock(256) for _ in range(9)] 
        )
        self.transpose = nn.Sequential(
            *[TransposeBlock(256//(2**index), 256//(2**(index+1))) for index in range(2)]
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1, stride=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        input = self.input_conv(x)
        input = self.conv(input)
        input = self.residual(input)
        input = self.transpose(input)
        input = self.output_conv(input)
        return x + input
    
class Discriminator(nn.Module):
    def __init__(self, num_chanels):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_chanels, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
        )
        self.linear = None
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        self.linear = nn.Linear(x.size(1), num_chanels)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    model = Generator()
    model.eval()
    
if __name__ == "main":
    main()
        
        
        