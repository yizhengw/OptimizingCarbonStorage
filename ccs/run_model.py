import numpy as np
import scipy.io as sio
import torch
import torch.fft

# c_rock, srw, src, pe, muw, xinj1, yinj1, xinj2, yinj2, xinj3, yinj3, xobs1, yobs1
Paremeters = np.load('Parameters.npy')

# permeability maps
mapcollections = np.load('mapcollections.npy')
# mapcollections = sio.loadmat('mapscollections.npy')['maps']


n = 1
# input
INJ_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t
K_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t
CROCK_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t
SRW_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t
SRC_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t
PE_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t
MUW_MAP = np.zeros((n, 80, 80, 8, 30)) # n, x, y, z, t


def map_normalize(x, mean, std):
    return (x-mean)/std

CROCK_mean, CROCK_std = 5.1213988098672205e-05, 1.7495288398342762e-05
SRW_mean, SRW_std = 0.24132516163828813, 0.052067532905783434
SRC_mean, SRC_std = 0.25620943989874656, 0.06011048467867877
PE_mean, PE_std = 11.10648700449306, 5.084056393056938
MUW_mean, MUW_std = 0.0007452311020595686, 8.551513629806743e-05


idx = 0
file_idx = 0

# input
xinj1, yinj1 = int(Paremeters[-7,file_idx]), int(Paremeters[-8,file_idx])
xinj2, yinj2 = int(Paremeters[-5,file_idx]), int(Paremeters[-6,file_idx])
xinj3, yinj3 = int(Paremeters[-3,file_idx]), int(Paremeters[-4,file_idx])
c_rock, srw, src, pe, muw = Paremeters[0,file_idx], Paremeters[1,file_idx], Paremeters[2,file_idx], Paremeters[3,file_idx], Paremeters[4,file_idx]
inj = np.zeros((80, 80, 8, 30))
inj[xinj1-1, yinj1-1, :, :] = 1
inj[xinj2-1, yinj2-1, :, 4:] = 1
inj[xinj3-1, yinj3-1, :, 9:] = 1
INJ_MAP[idx,:,:,:,:] = inj
CROCK_MAP[idx,:,:,:,:] = c_rock
SRW_MAP[idx,:,:,:,:] = srw
SRC_MAP[idx,:,:,:,:] = src
PE_MAP[idx,:,:,:,:] = pe
MUW_MAP[idx,:,:,:,:] = muw

K_MAP[idx,:,:,:,:] = mapcollections[file_idx,...,np.newaxis]

CROCK_MAP = map_normalize(CROCK_MAP, CROCK_mean, CROCK_std)
SRW_MAP = map_normalize(SRW_MAP, SRW_mean, SRW_std)
SRC_MAP = map_normalize(SRC_MAP, SRC_mean, SRC_std)
PE_MAP = map_normalize(PE_MAP, PE_mean, PE_std)
MUW_MAP = map_normalize(MUW_MAP, MUW_mean, MUW_std)

grid_x = np.linspace(0, 1, 80)
grid_x = grid_x[np.newaxis,:,np.newaxis,np.newaxis, np.newaxis]

grid_y = np.linspace(0, 1, 80)
grid_y = grid_y[np.newaxis,np.newaxis,:,np.newaxis, np.newaxis]

grid_z = np.linspace(0, 0.1, 8)
grid_z = grid_z[np.newaxis,np.newaxis,np.newaxis,:, np.newaxis]

grid_t = np.load(f'sim_1/time_days.npy')[1:31,0]
grid_t /= np.max(grid_t)
grid_t = grid_t[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]


x = np.zeros((n, 80, 80, 8, 30, 11))
x[...,0] = K_MAP
x[...,1] = INJ_MAP
x[...,2] = CROCK_MAP
x[...,3] = SRW_MAP
x[...,4] = SRC_MAP
x[...,5] = PE_MAP
x[...,6] = MUW_MAP

x[...,7] = grid_x
x[...,8] = grid_y
x[...,9] = grid_z
x[...,10] = grid_t

x = x.astype(np.float32)
x = torch.from_numpy(x)
print(x.shape)


from gas_saturation_injection_period import *
model = torch.load('../models/sg_inj_model.pt')

def dnorm_inj_sg(x):
    return x*4322407633.652639 + 2181250000.0
    
    
device = torch.device('cuda:0')

xx = x.to(device)

torch.cuda.empty_cache()

def rn(model, xx):
    with torch.no_grad():
        sg_pred, mass_pred = model(xx)
    sg_pred, mass_pred
    
    
 
# sg_pred, mass_pred = model(xx)

x_plot = xx.cpu().detach().numpy()
pred_plot = sg_pred.cpu().detach().numpy() # Full field prediction
mass_plot = dnorm_inj_sg(mass_pred.cpu().detach().numpy().transpose())

plt.figure(figsize=(15,3))
plt.jet()
pred_plot.shape
for it, t in enumerate([0,5,10,15,20,25]):        
    plt.subplot(1,6,it+1)
    plt.title(f'pred, t ={t}')
    plt.imshow(pred_plot[0,:,:,-1,t].squeeze())
    plt.clim([0,0.7])
    plt.colorbar(fraction=0.02)

plt.tight_layout()
plt.show()

plt.figure(figsize=(5,3))

color = ['red', 'orange', 'yellow', 'green', 'deeppink', 'black' , 'blue', 'indigo']
for line in range(8):
    plt.plot(mass_plot[:,line], c=color[line], linestyle='-')

plt.xlabel('time')

plt.ylabel('mass')
plt.show()