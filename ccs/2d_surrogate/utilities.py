import numpy as np

m_to_grid = lambda a : int(a/2.08333)
dnorm_inj = lambda a : (a  * (3e6 - 3e5) + 3e5) / (1e6 /365*1000/1.862) 
dnorm_temp = lambda a : a * (180 - 30) + 30
dnorm_P = lambda a : a * (300 - 100) + 100
dnorm_lam = lambda a : a * 0.4 + 0.3
dnorm_Swi = lambda a : a * 0.2 + 0.1

norm_inj = lambda a : (a * (1e6 /365*1000/1.862)  - 3e5) / (3e6 - 3e5)
norm_temp = lambda a : (a - 30) / (180 - 30)
norm_P = lambda a : (a - 100) / (300 - 100)
norm_lam = lambda a : (a - 0.3) / 0.4
norm_Swi = lambda a : (a - 0.1) / 0.2

def make_input(b, k, perf, inj_rate, temp, P, Swi, lam):
    k[m_to_grid(b):,:] = 0.0000001 # very small permeability
    k_map = np.log(k).reshape((1, 96, 200)) / 15
    k_map = np.repeat(k_map, 18, axis=0)
    
    perf_map = np.zeros((1, 96, 200))
    top = m_to_grid(b-perf[0])
    btm = m_to_grid(b-perf[1])
    perf_map[0, top:btm, 0] = 1
    inj_map = np.ones((1, 96, 200)) * norm_inj(inj_rate)
    temp_map = np.ones((1, 96, 200)) * norm_temp(temp)
    P_map = np.ones((1, 96, 200)) * norm_P(P)
    Swi_map = np.ones((1, 96, 200)) * norm_Swi(Swi)
    lam_map = np.ones((1, 96, 200)) * norm_lam(lam)

    x = np.concatenate((k_map, perf_map, inj_map, temp_map, P_map, Swi_map, lam_map), axis=0).transpose((1,2,0))
    return x[np.newaxis,...,np.newaxis]