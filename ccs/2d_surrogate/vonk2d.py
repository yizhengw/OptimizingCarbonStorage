import numpy as np
import math
import time
from scipy.interpolate import griddata
from vonk2d_nonuni import *

def generateSyntheticField(rseed, nx, nz, dx, dz, ix, iz, ax, az, pop, med, nu, vel, frac):

    # Initialize variables and inputs
    # This program is stochastic - it will generate a different random map each time, even for the same inputs
    """
    rseed = np.random.randint(10000, size=1)
    nx = 27826
    nz = 96
    dx = 1
    dz = 1
    ix = nx*dx
    iz = nz*dz
    ax = 10
    az = 20
    pop = 1 # 1 = gaussian; 2 = PDF
    med = 3 # 1 = gaussian; 2 = exponential; 3 = von karman
    nu = 0.1 # only applies to von karman
    vel = [100,200,300]
    frac = [0.1,0.2,0.3]
    """

    # Create the permeability map, which is 96 x 27826 (uniformly spaced)
    start = time.time()
    data = vonk2d_nonuni(rseed,dx,dz,ax,az,ix,iz,pop,med,nu,vel,frac)
    end = time.time()
    # print("Time to generate uniform grid data:", np.round(end-start, 5), "sec")
    #print(data.shape)

    # Re-grid the data for the CCSNet format, which is 96 x 200 (nonuniformly spaced)
    y_uni = np.linspace(0,200,96)
    x_uni = np.linspace(3.5938,27826*3.5938,27826)
    x_uni, y_uni = np.meshgrid(x_uni, y_uni)

    x_irr = np.cumsum(1.035012 ** np.arange(200) * 3.5938)
    y_irr = np.linspace(0,200,96)
    x_irr, y_irr = np.meshgrid(x_irr, y_irr)

    start = time.time()
    data_gridded = griddata((x_uni.flatten(), y_uni.flatten()), data.flatten(), (x_irr, y_irr), method='nearest')
    end = time.time()
    # print("Time to interpolate:", np.round(end-start, 5), "sec")
    #print(data_gridded.shape)

    # Mask the data for pop = 2
    if pop==2:
        start = time.time()
        frac=np.cumsum(frac/np.sum(frac)) # normalize and cumsum
        sdata=np.sort(data.flatten())
        nn=nx*nz

        # Calculate critical values in randdata to resemble fraction
        fraclimit=np.zeros((len(frac)+1,1))
        fraclimit[0]=sdata[0]

        for n in range(len(frac)):
            fraclimit[n+1] = sdata[int(np.round(nn*frac[n]) - 1)]

        fraclimit[0] = fraclimit[0]- 0.1*np.abs(fraclimit[0])
        fraclimit[-1] = fraclimit[-1] + 0.1*np.abs(fraclimit[-1])

        rdata=np.zeros((np.size(data_gridded, 0), np.size(data_gridded, 1)))

        for n in range(len(frac)):
            for r in range(rdata.shape[0]):
                for c in range(rdata.shape[1]):
                    mask = (data_gridded[r, c]>fraclimit[n]) and (data_gridded[r, c]<=fraclimit[n+1])
                    rdata[r, c] = rdata[r, c] + mask * vel[n]
        data_gridded=rdata
        end = time.time()
        # print("Time to mask:", np.round(end-start, 5), "sec")

    return data_gridded

def kmap_generation(rseed, k_mean, k_std, az, ax, med, pop, nMaterials):
    if med == 'Gaussian':
        med = 1
    elif med == 'Von Karman':
        med = 3
    if pop == 'Continuous':
        pop = 1
    elif pop == 'Discontinuous':
        pop = 2

    vel = np.random.random(size=(nMaterials,))
    if pop == 2:
        vel = k_mean + (vel - np.mean(vel)) * (k_std/np.std(vel))
    frac = np.ones((nMaterials,))

    nx = 27826
    nz = 96
    dx = 1
    dz = 1
    ix = nx*dx
    iz = nz*dz
    nu = 0.6 # only applies to von karman

    kmap = generateSyntheticField(rseed, nx, nz, dx, dz, ix, iz, int(ax/3.5938), int(az/(200/96)), 
                                  pop, med, nu, vel, frac)

    if pop == 1:
            kmap = k_mean + (kmap - np.mean(kmap)) * (k_std/np.std(kmap))

    kmap[kmap <= 0.001] = 0.001

    return kmap