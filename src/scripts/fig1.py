import paths
import utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import astropy.units as u
import astropy.constants as c
import cmasher as cmr
from scipy.integrate import trapezoid, cumulative_trapezoid


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 14


# update various fontsizes to match
params = {'figure.figsize': (6,4),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.7 * fs,
          'ytick.labelsize': 0.7 * fs}
plt.rcParams.update(params)
cs2 = cmr.take_cmap_colors('cmr.emerald', 3, cmap_range=(0.15, 0.9), return_fmt='hex')

# set up the LISA frequency grid
f_LISA = np.logspace(-1, -5, 15) * u.Hz

# set up the LIGO eccentricity range
e_LIGO_lo = np.logspace(-6, -4, 10)
e_LIGO_hi = np.linspace(0.0002, 0.001, 5)
e_LIGO = np.append(e_LIGO_lo, e_LIGO_hi)
e_LIGO = np.append(0, e_LIGO)
e_LIGO_round = np.array([f"{e:.2e}" for e in e_LIGO])

# get the mass, mass ratio, and rate grids and meshgrids
down_samp_fac_q=75
down_samp_fac_m1=75

# indices to select for plotting
ecc_ind = 0
q_ind = 6
ind_m_10 = 1
ind_m_35 = 5
ind_m_80 = 11

# mass_1, M1, dN_dm1dqdVcdt comes out with units
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = utils.get_LIGO_rate(down_samp_fac_m1=down_samp_fac_m1, down_samp_fac_q=down_samp_fac_q)
dN_dm1dVcdt = trapezoid(dN_dm1dqdVcdt, mass_ratio, axis=0)

MM, QQ, EE_LIGO, FF = np.meshgrid(mass_1, mass_ratio, e_LIGO, f_LISA, indexing='ij')

EE_LISA = np.load(paths.data / 'e_LISA.npy')
TT_LISA = np.load(paths.data / 't_LIGO.npy') * u.Gyr
VC = np.load(paths.data / 'comoving_volume.npy') * u.Gpc**3
DH = np.load(paths.data / 'horizon_dist.npy') * u.Gpc 

# get time to chirp back from 10 Hz to LISA frequency FF
dT_LIGO_df_LISA = utils.dTmerger_df(MM, QQ*MM, FF, EE_LISA).to(u.yr / u.Hz)

# calculate dn_LISA as a function of primary mass and LISA frequency
# for circular binaries
dn_LISA = np.zeros((len(mass_1), len(f_LISA)))
for ff, forb in enumerate(f_LISA):
    dn_LISA[:,ff] = trapezoid(dT_LIGO_df_LISA[:,:,ecc_ind,ff] * dN_dm1dqdVcdt[:, :].T, mass_ratio).to(1/u.Hz / u.Msun / u.Gpc**3)
#dn_LISA = dn_LISA * u.s / u.Msun / u.Gpc**3
    
dN_dm1_list = []
for snr_thresh in [1,7,12]:
    VC_new = utils.get_VC_new(snr_thresh, DH, VC)
    dN_dm1 = np.zeros(len(mass_1)) / u.Msun
    for ii, m in enumerate(mass_1):
        dN_dm1dq = np.zeros(len(mass_ratio)) / u.Msun
        import pdb; pdb.set_trace()

        for jj, q in enumerate(mass_ratio):
            dN_dm1dq[jj] = trapezoid(dN_dm1dqdVcdt[jj,ii] * -dT_LIGO_df_LISA[ii,jj,ecc_ind,:]* VC_new[ii,jj,ecc_ind,:], f_LISA).to(1/u.Msun)
        dN_dm1[ii] = trapezoid(dN_dm1dq, mass_ratio)
        
    dN_LISA_obs = np.zeros((len(mass_1), len(f_LISA)))
    
    dN_LISA_obs[:,ff] = trapezoid(dT_LIGO_df_LISA[:,:,ecc_ind,ff] * dN_dm1dqdVcdt[:, :].T * VC_new[:,:,ecc_ind,ff], mass_ratio).to(1/u.Hz / u.Msun)
    dN_LISA_obs = dN_LISA_obs * u.s / u.Msun
    dN_dm1_list.append(dN_dm1)


labels = [r'$M_1=10\,M_{\odot}$', r'$M_1=35\,M_{\odot}$', r'$M_1=80\,M_{\odot}$']
labels_snr = [r'$SNR > 1$', r'$SNR > 7$', r'$SNR > 12$']



fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,3))


ax1.plot(mass_1.value, dN_dm1dVcdt.value, ls='-', color='black')
ax1.fill_between(mass_1.value[7:10], np.ones(len(mass_1.value[4:7]))*1e-7, dN_dm1dVcdt.value[7:10], color=cs2[0], alpha=0.3, label='$M_{\mathrm{BH,1}}$=10\,$M_{\odot}$')
ax1.fill_between(mass_1.value[33:36], np.ones(len(mass_1.value[21:24]))*1e-7, dN_dm1dVcdt.value[33:36], color=cs2[1], alpha=0.3, label='$M_{\mathrm{BH,1}}$=35\,$M_{\odot}$')
ax1.fill_between(mass_1.value[79:82], np.ones(len(mass_1.value[52:55]))*1e-7, dN_dm1dVcdt.value[79:82], color=cs2[2], alpha=0.3, label='$M_{\mathrm{BH,1}}$=80\,$M_{\odot}$')

ax1.set_yscale('log')
ax1.set_ylim(1e-4, 8)
ax1.minorticks_on()
ax1.tick_params(axis='both', labelsize=12)
ax1.set_ylabel(r'$\mathrm{d}N/ \mathrm{d}M_{\mathrm{BH,1}}\mathrm{d}V_c\mathrm{d}t$ [$\rm{Gpc}^{-3}\,M_{\odot}^{-1}\,\rm{yr}^{-1}$]', size=14)
ax1.set_xlabel(r'$M_{\mathrm{BH,1}}$ [$M_{\odot}$]', size=14)
ax1.legend(prop={'size':14}, frameon=False)


for ii, ind in enumerate([ind_m_10, ind_m_35, ind_m_80]):
    ax2.plot(FF[ind,q_ind,ecc_ind,:], dn_LISA.value[ind,:], color=cs2[ii], label=labels[ii])
ax2.set_xscale('log')
ax2.set_xlabel(r'$f_{\rm orb}$ [Hz]')
ax2.set_ylabel(r'$\mathrm{d}^3N / \mathrm{d}M_{\mathrm{BH,1}} \mathrm{d}f_{\rm orb}$ [$\rm{Gpc}^{-3}\,M_{\odot}^{-1}\,\rm{Hz}^{-1}$]')
ax2.set_yscale('log')
ax2.set_xlim(5e-5, 0.1)
ax2.set_ylim(1e-7, 1e15)
ax2.legend(frameon=False)
ax2.minorticks_on()

for ii, ind in enumerate([ind_m_10, ind_m_35, ind_m_80]):
    ax3.plot(f_LISA, DH[ind, q_ind, ecc_ind, :], color=cs2[ii], label=labels[ii])
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend(frameon=False)
ax3.set_ylabel(r'$D_{\rm{horizon,\,SNR>12}}\,[\rm{Gpc}]$')
ax3.set_xlabel(r'$f_{\rm orb}$ [Hz]')
ax3.set_ylim(1e-5, 6)
ax3.set_xlim(5e-5, 0.1)


fig.tight_layout()
#fig.savefig(paths.figures / 'fig1.pdf', dpi=150, facecolor='white')
fig.savefig('fig1.pdf', dpi=150, facecolor='white')