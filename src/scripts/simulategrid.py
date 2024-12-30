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
f_LISA = np.logspace(-1, -5, 150) * u.Hz

# set up the LIGO eccentricity range
e_LIGO = np.logspace(-6, np.log10(0.001), 20)
e_LIGO = np.append(0, e_LIGO)
e_LIGO_round = np.array([f"{e:.2e}" for e in e_LIGO])

# get the mass, mass ratio, and rate grids and meshgrids
down_samp_fac_q=30
down_samp_fac_m1=15
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = utils.get_LIGO_rate(down_samp_fac_m1=down_samp_fac_m1, down_samp_fac_q=down_samp_fac_q)

MM, QQ, EE_LIGO, FF = np.meshgrid(mass_1, mass_ratio, e_LIGO, f_LISA, indexing='ij')

# get the eccentricity mappings and time to evolve between each eccentricity
dat_in = list(zip(EE_LIGO.flatten(), FF.flatten(), MM.flatten(), QQ.flatten()*MM.flatten()))
with MultiPool(processes=nproc) as pool:
    dat_out = list(tqdm.tqdm(pool.imap(utils.get_e_LISA_t_LIGO, dat_in), total=len(dat_in)))
EE_LISA, TT_LIGO = zip(*dat_out)
EE_LISA = np.array(EE_LISA).reshape(FF.shape)
TT_LIGO = np.array(TT_LIGO).reshape(FF.shape) * u.yr

np.save(paths.data / 't_LIGO', TT_LIGO.value)
np.save(paths.data / 'e_LISA', EE_LISA)


# get the horizon distances and comoving volumes
num_chunks = 10
snr_thresh = 12
dat_in = list(zip(MM.flatten(), QQ.flatten(), EE_LISA.flatten(), FF.flatten(), snr_thresh * np.ones(len(MM.flatten()))))

chunked_list = utils.chunk_list(dat_in, num_chunks)
dat_list = []
for ii, chunk in enumerate(chunked_list):
    print('running chunk: ' + str(ii))
    with MultiPool(processes=nproc) as pool:
        dat_out = list(tqdm.tqdm(pool.imap(utils.get_Vc_Dh, chunk), total=len(chunk)))
        dat_list.extend(dat_out)
DH, VC = zip(*dat_list)
DH = np.array(DH).reshape(QQ.shape) * u.Gpc
VC = np.array(VC).reshape(QQ.shape) * u.Gpc**3

np.save(paths.data / f'comoving_volume', VC.value)
np.save(paths.data / f'horizon_distance', DH.value)

print('all done friend!')
