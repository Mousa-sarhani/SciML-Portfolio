import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# 1. Fourier Layer Definition
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# 2. Operator Architecture
class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.lifting = nn.Conv2d(1, self.width, 1)
        
        self.fourier1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        
        self.fourier2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        
        self.fourier3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.projection = nn.Conv2d(self.width, 1, 1)
        
    def forward(self, x):
        x = self.lifting(x)
        x = F.gelu(self.fourier1(x) + self.w1(x))
        x = F.gelu(self.fourier2(x) + self.w2(x))
        x = F.gelu(self.fourier3(x) + self.w3(x))
        x = self.projection(x)
        return x

# 3. Resolution Invariance Test
model = FNO2d(modes1=12, modes2=12, width=32)

print("--- Low Resolution Test ---")
low_res_tensor = torch.exp(torch.randn(5, 1, 32, 32))
low_res_output = model(low_res_tensor)
print(f"Permeability Input:  {low_res_tensor.shape}")
print(f"Pressure Output:     {low_res_output.shape}\n")

print("--- High Resolution Test ---")
high_res_tensor = torch.exp(torch.randn(5, 1, 128, 128))
high_res_output = model(high_res_tensor)
print(f"Permeability Input:  {high_res_tensor.shape}")
print(f"Pressure Output:     {high_res_output.shape}")