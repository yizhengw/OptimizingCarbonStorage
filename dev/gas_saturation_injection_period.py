import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import math
import gc
import h5py

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()

        """
        4D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4,-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4], self.weights4)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights5)
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4], self.weights6)
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4], self.weights7)
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4], self.weights8)        

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

    
class volume_to_scalar(nn.Module):
    def __init__(self):
        super(volume_to_scalar, self).__init__()
        self.cnn0 = nn.Conv3d(30, 60, (2, 2, 2), stride = 2)
        self.cnn1 = nn.Conv3d(60, 120, (2, 2, 2), stride = 2)
        self.cnn2 = nn.Conv3d(120, 120, (2, 2, 2), stride = 2)
        self.cnn3 = nn.Conv3d(120, 120, (2, 2, 1), stride = 2)
        self.cnn4 = nn.Conv3d(120, 60, (2, 2, 1), stride = 2)
        self.cnn5 = nn.Conv3d(60, 30, (2, 2, 1), stride = 2)
        
    def forward(self, x):
#         print('SG', x.shape)
        batchsize = x.shape[0]
        x = self.cnn0(x)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x).view(batchsize, 30)
#         print('m', x.shape)
        return x


class SimpleBlock4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width):
        super(SimpleBlock4d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.fc0 = nn.Linear(11, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv1 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv2 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv3 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv4 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)
        self.mass1 = volume_to_scalar()
        self.mass2 = volume_to_scalar()
        self.mass3 = volume_to_scalar()
        self.mass4 = volume_to_scalar()
        self.mass5 = volume_to_scalar()
        self.mass6 = volume_to_scalar()
        self.mass7 = volume_to_scalar()
        self.mass8 = volume_to_scalar()
        
    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z, size_t = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2 
        x = nn.GELU()(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2 
        x = nn.GELU()(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2 
        x = nn.GELU()(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2 
        x = nn.GELU()(x)

        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        x = x1 + x2 
        x = nn.GELU()(x)
        
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        SG = torch.clone(x)
        
        x = x.view(batchsize, 80, 80, 8, 30)
        x = x.permute(0,4,1,2,3)
        mass1 = self.mass1(x)
        mass2 = self.mass2(x)
        mass3 = self.mass3(x)
        mass4 = self.mass4(x)        
        mass5 = self.mass5(x)
        mass6 = self.mass6(x)
        mass7 = self.mass7(x)
        mass8 = self.mass8(x)
        mass = torch.cat((mass1, mass2, mass3, mass4, mass5, mass6, mass7, mass8), axis=0)
        
        return [SG, mass]
    
class Net4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width):
        super(Net4d, self).__init__()

        """
        A wrapper function
        """
        self.conv1 = SimpleBlock4d(modes1, modes2, modes3, modes4, width)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z, size_t = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        SG, mass = self.conv1(x)
        return SG, mass


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c