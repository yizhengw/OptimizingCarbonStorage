import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import operator
from functools import reduce
from functools import partial

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
    
class SimpleBlock4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width):
        super(SimpleBlock4d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.fc0 = nn.Linear(11, self.width)

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
        
    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z, size_t = x.shape[1]+8, x.shape[2]+8, x.shape[3]+8, x.shape[4]+8
        
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        x = F.pad(x, [4, 4, 4, 4, 4, 4, 4, 4])

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
        
        x = x[:, :, 4:-4, 4:-4,  4:-4, 4:-4]
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)        
        x = x.view(batchsize, 80, 80, 8, 20)
        
        return x

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
        SG = self.conv1(x)
        return SG

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


class volume_to_scalar_post(nn.Module):
    def __init__(self):
        super(volume_to_scalar_post, self).__init__()
        self.cnn0 = nn.Conv3d(20, 20, (2, 2, 2), stride = 2)
        self.cnn1 = nn.Conv3d(20, 20, (2, 2, 2), stride = 2)
    
        self.bncnn0 = nn.BatchNorm3d(20)
        self.bncnn1 = nn.BatchNorm3d(20)
    
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        
    def forward(self, x):
        batchsize = x.shape[0]
        
        x = self.cnn0(x)
        x = self.bncnn0(x)
        x = nn.GELU()(x)
        
        x = self.cnn1(x)
        x = self.bncnn1(x)
        x = nn.GELU()(x)

        x = x.view(batchsize, -1)
        
        return  x

class sg_to_mass_post(nn.Module):
    def __init__(self):
        super(sg_to_mass_post, self).__init__()
        self.sg = volume_to_scalar_post()

        self.fc11 = nn.Linear(16000, 100)
        self.fc12 = nn.Linear(105, 20)
        
        self.fc21 = nn.Linear(16000, 100)
        self.fc22 = nn.Linear(105, 20)
        
        self.fc31 = nn.Linear(16000, 100)
        self.fc32 = nn.Linear(105, 20)

    def forward(self, x, param):
        x = x.permute(0,4,1,2,3)
        sg = self.sg(x)
        
        mass_1 = self.fc11(sg)
        mass_1 = F.relu(mass_1)
        mass_1 = torch.cat([mass_1, param], axis=-1)
        mass_1 = self.fc12(mass_1)
        mass_1 = F.relu(mass_1)
        
        mass_2 = self.fc21(sg)
        mass_2 = F.relu(mass_2)
        mass_2 = torch.cat([mass_2, param], axis=-1)
        mass_2 = self.fc22(mass_2)
        mass_2 = F.relu(mass_2)
        
        mass_3 = self.fc31(sg)
        mass_3 = F.relu(mass_3)
        mass_3 = torch.cat([mass_3, param], axis=-1)
        mass_3 = self.fc32(mass_3)
        mass_3 = F.relu(mass_3)
        
        mass = torch.cat((mass_1, mass_2, mass_3), axis=0)
        return mass
    
    
class sg_inj_post(nn.Module):
    def __init__(self, sg_model, mass_model):
        super(sg_inj_post, self).__init__()
        self.sg_model = sg_model
        self.mass_model = mass_model

    def forward(self, x, param):
        sg_pred = self.sg_model(x)
        sg_pred = sg_pred * 0.11579982665912925 + 0.020176139003292584
        x_mass = ((sg_pred - 0.020404457779112768)/ 0.11628368029910742)
        mass_pred = self.mass_model(x_mass, param)
        return sg_pred, mass_pred