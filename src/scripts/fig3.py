import paths
import utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import astropy.units as u
import astropy.constants as c
import legwork as lw

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 12


# update various fontsizes to match
params = {'figure.figsize': (6,4),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.7 * fs,
          'ytick.labelsize': 0.7 * fs}
plt.rcParams.update(params)



q_ind = 9
ind_m_10 = 3
ind_m_35 = 11
ind_m_80 = 24

labels = [r'$M_1=10\,M_{\odot}$', r'$M_1=35\,M_{\odot}$', r'$M_1=80\,M_{\odot}$']
labels_snr = [r'$SNR > 1$', r'$SNR > 7$', r'$SNR > 12$']

# set up the LISA frequency grid
f_LISA = np.logspace(-1, -5, 150) * u.Hz

# set up the LIGO eccentricity range
e_LIGO = np.logspace(-6, np.log10(0.001), 20)
e_LIGO = np.append(0, e_LIGO)
e_LIGO_round = np.array([f"{e:.2e}" for e in e_LIGO])

# get the mass, mass ratio, and rate grids and meshgrids
down_samp_fac_q=10
down_samp_fac_m1=10
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = utils.get_LIGO_rate(down_samp_fac_m1=down_samp_fac_m1, down_samp_fac_q=down_samp_fac_q)

MM, QQ, EE_LIGO, FF = np.meshgrid(mass_1, mass_ratio, e_LIGO, f_LISA, indexing='ij')

EE_LISA = np.load(paths.data / 'e_LISA.npy')
TT_LISA = np.load(paths.data / 't_LIGO.npy') * u.Gyr
VC = np.load(paths.data / 'comoving_volume.npy') * u.Gpc**3
DH = np.load(paths.data / 'horizon_dist.npy') * u.Gpc 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

for e_ind, ls in zip([1, 9, 19], ['-', ':', '-.']):
    m_c = lw.utils.chirp_mass(MM[ind_m_35,q_ind,e_ind,:], MM[ind_m_35,q_ind,e_ind,:]*QQ[ind_m_35,q_ind,e_ind,:])
    chirp = lw.utils.fn_dot(m_c, FF[ind_m_35,q_ind,e_ind,:], EE_LISA[ind_m_35,q_ind,e_ind,:], 1)
    ax1.plot(f_LISA.value, DH[ind_m_35,q_ind,e_ind,:].value, ls=ls, color='navy', label=f'e at fLIGO = {e_LIGO_round[e_ind]}')
    ax2.plot(FF[ind_m_35,q_ind,e_ind,:], np.abs(chirp), ls=ls, color='black', label=f'e at fLIGO = {e_LIGO_round[e_ind]}')
    
ax1.set_ylabel('horizon distance [Gpc]')
ax2.set_ylabel('orbital frequency evolution [Hz/yr]')
ax2.set_xlabel('orbital frequency [Hz]')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.legend(loc='lower right')
fig.tight_layout()


plt.savefig(paths.figures / 'fig2.pdf', dpi=150, facecolor='white')