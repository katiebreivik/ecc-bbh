import numpy as np
import legwork as lw
import astropy.units as u
import deepdish as dd
from schwimmbad import MultiPool
import tqdm
import utils
import paths


# number of processes to use
nproc = 96

# set up the LISA frequency grid
f_LISA = np.logspace(-1, -5, 15)

# set up the LIGO eccentricity range
e_LIGO_lo = np.logspace(-6, -4, 10)
e_LIGO_hi = np.linspace(0.0002, 0.001, 5)
e_LIGO = np.append(e_LIGO_lo, e_LIGO_hi)
e_LIGO = np.append(0, e_LIGO)
e_LIGO_round = np.array([f"{e:.2e}" for e in e_LIGO])

# get the mass, mass ratio, and rate grids and meshgrids
down_samp_fac_q=75
down_samp_fac_m1=75
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = utils.get_LIGO_rate(down_samp_fac_m1=down_samp_fac_m1, down_samp_fac_q=down_samp_fac_q)

MM, QQ, EE_LIGO, FF = np.meshgrid(mass_1, mass_ratio, e_LIGO, f_LISA, indexing='ij')

# get the eccentricity mappings and time to evolve between each eccentricity
dat_in = list(zip(EE_LIGO.flatten(), FF.flatten(), MM.flatten(), QQ.flatten()*MM.flatten()))
with MultiPool(processes=nproc) as pool:
    dat_out = list(pool.imap(utils.get_e_LISA_t_LIGO, dat_in))
print('getting e_LISA and t_LIGO')
EE_LISA, TT_LIGO = zip(*dat_out)
#EE_LISA, TT_LIGO = utils.get_e_LISA_t_LIGO(list(zip(EE_LIGO.flatten(), FF.flatten(), MM.flatten(), QQ.flatten()*MM.flatten())))
EE_LISA = np.array(EE_LISA).reshape(FF.shape)
TT_LIGO = np.array(TT_LIGO).reshape(FF.shape) * u.yr

np.save(paths.data / 't_LIGO', TT_LIGO.value)
np.save(paths.data / 'e_LISA', EE_LISA)

print('getting horizon distance and comoving volume')
# get the horizon distances and comoving volumes
num_chunks = 10
snr_thresh = 12
dat_in = list(zip(MM.flatten()*u.Msun, QQ.flatten(), EE_LISA.flatten(), FF.flatten()*u.Hz, snr_thresh * np.ones(len(MM.flatten()))))

chunked_list = utils.chunk_list(dat_in, num_chunks)
dat_list = []
for ii, chunk in enumerate(chunked_list):
    #for c in chunk:
    #    d = utils.get_Vc_Dh(c)
    #    dat_list.append(d)
    
    print('running chunk: ' + str(ii))
    with MultiPool(processes=nproc) as pool:
        dat_out = list(pool.imap(utils.get_Vc_Dh, chunk))
        dat_list.extend(dat_out)
    
DH, VC = zip(*dat_list)
DH = np.array(DH).reshape(QQ.shape) * u.Gpc
VC = np.array(VC).reshape(QQ.shape) * u.Gpc**3

np.save(paths.data / 'horizon_dist', DH.value)
np.save(paths.data / 'comoving_volume', VC.value)

print('all done friend!')
