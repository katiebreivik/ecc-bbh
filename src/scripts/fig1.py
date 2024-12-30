import paths
import utils

from bilby.core.result import read_in_result
import deepdish as dd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import astropy.units as u
import astropy.constants as c
import cmasher as cmr
from scipy.integrate import trapezoid, cumulative_trapezoid


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = True
fs = 12


# update various fontsizes to match
params = {'figure.figsize': (6,4),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.7 * fs,
          'ytick.labelsize': 0.7 * fs}
plt.rcParams.update(params)


cs = cmr.take_cmap_colors('cmr.sapphire', len(mass_1), cmap_range=(0.15, 0.9), return_fmt='hex')
cs2 = cmr.take_cmap_colors('cmr.emerald', 3, cmap_range=(0.15, 0.9), return_fmt='hex')
ecc_ind = 0
q_ind = 16
ind_m_10 = 3
ind_m_35 = 14
ind_m_80 = 32

labels = [r'$M_1=10\,M_{\odot}$', r'$M_1=35\,M_{\odot}$', r'$M_1=80\,M_{\odot}$']
labels_snr = [r'$SNR > 1$', r'$SNR > 7$', r'$SNR > 12$']

# set up the LISA frequency grid
f_LISA = np.logspace(-1, -5, 150) * u.Hz

# set up the LIGO eccentricity range
e_LIGO = np.logspace(-6, np.log10(0.001), 20)
e_LIGO = np.append(0, e_LIGO)
e_LIGO_round = np.array([f"{e:.2e}" for e in e_LIGO])

# get the mass, mass ratio, and rate grids
down_samp_fac_q=30
down_samp_fac_m1=15
mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt = utils.get_LIGO_rate(down_samp_fac_m1=down_samp_fac_m1, down_samp_fac_q=down_samp_fac_q)

MM, QQ, EE_LIGO, FF = np.meshgrid(mass_1, mass_ratio, e_LIGO, f_LISA, indexing='ij')

EE_LISA = np.load(paths.data / 'e_LISA.npy')
TT_LISA = np.load(paths.data / 't_merge.npy') * u.Gyr
VC = np.load(paths.data / 'comoving_volume.npy') * u.Gpc**3
DH = np.load(paths.data / 'horizon_distance.npy') * u.Gpc 


dn_LISA = np.zeros((len(mass_1), len(f_LISA)))
for ff, forb in enumerate(f_LISA):
    dn_LISA[:,ff] = trapezoid(dT_LIGO_df_LISA[:,:,ecc_ind,ff] * dN_dm1dqdVcdt[:, :].T, mass_ratio).to(1/u.Hz / u.Msun / u.Gpc**3)

dN_dm1_list = []
for snr_thresh in [1,7,12]:
    VC_new = get_VC_new(snr_thresh, DH, VC)
    dN_dm1 = np.zeros(len(mass_1)) / u.Msun
    for ii, m in enumerate(mass_1):
        dN_dm1dq = np.zeros(len(mass_ratio)) / u.Msun
        for jj, q in enumerate(mass_ratio):
            dN_dm1dq[jj] = trapezoid(dN_dm1dqdVcdt[jj,ii] * -dT_LIGO_df_LISA[ii,jj,ecc_ind,:]* VC_new[ii,jj,ecc_ind,:], f_LISA).to(1/u.Msun)
        dN_dm1[ii] = trapezoid(dN_dm1dq, mass_ratio)
        
    dN_LISA_obs = np.zeros((len(mass_1), len(f_LISA)))
    
    dN_LISA_obs[:,ff] = trapezoid(dT_LIGO_df_LISA[:,:,ecc_ind,ff] * dN_dm1dqdVcdt[:, :].T * VC_new[:,:,ecc_ind,ff], mass_ratio).to(1/u.Hz / u.Msun)
    dN_LISA_obs = dN_LISA_obs * u.s / u.Msun
    dN_dm1_list.append(dN_dm1)
dn_LISA = dn_LISA * u.s / u.Msun / u.Gpc**3
dn_dm1 = trapezoid(f_LISA, dn_LISA).to(1/u.Msun / u.Gpc**3)
dn_dm1[0] = 1e-19 / u.Msun / u.Gpc**3

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
#p = ax1.scatter(FF[:,q_ind,ecc_ind,:], MM[:,q_ind,ecc_ind,:], c=dn_LISA.value, norm=colors.LogNorm(vmax=1e14), s=14, marker='o')
for ii, ind in enumerate([ind_m_10, ind_m_35, ind_m_80]):
    ax1.plot(FF[ind,q_ind,ecc_ind,:], dn_LISA.value[ind,:], color=cs[ind], label=labels[ii])
ax1.set_xscale('log')
#cbar = plt.colorbar(p)
#cbar.set_label(label=r'$\mathrm{d}^3N / \mathrm{d}M_{\mathrm{BH,1}} \mathrm{d}f_{\rm orb}$ [$\rm{Gpc}^{-3}\,M_{\odot}^{-1}\,\rm{Hz}^{-1}$]')

ax1.set_xlabel(r'$f_{\rm orb}$ [Hz]')
ax1.set_ylabel(r'$\mathrm{d}^3N / \mathrm{d}M_{\mathrm{BH,1}} \mathrm{d}f_{\rm orb}$ [$\rm{Gpc}^{-3}\,M_{\odot}^{-1}\,\rm{Hz}^{-1}$]')
ax1.set_yscale('log')
ax1.set_xlim(5e-5, 0.1)
ax1.set_ylim(1e-7, 1e15)
ax1.legend()
ax1.minorticks_on()

for ii, ind in enumerate([ind_m_10, ind_m_35, ind_m_80]):
    ax2.plot(f_LISA, DH[ind, q_ind, ecc_ind, :], color=cs[ind], label=labels[ii])
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()
ax2.set_ylabel(r'$D_{\rm{horizon,\,SNR>12}}\,[\rm{Gpc}]$')
ax2.set_xlabel(r'$f_{\rm orb}$ [Hz]')
ax2.set_ylim(1e-5, 6)
ax2.set_xlim(5e-5, 0.1)

ax3.plot(mass_1.value, dn_dm1.value, ls='--', color='black')
ax3.set_yscale('log')
ax3.set_ylim(1e5, 5e10)
ax3.minorticks_on()
ax3.set_ylabel(r'$\mathrm{d}N_{\rm{intrinsic}} / \mathrm{d}M_{\mathrm{BH,1}}$ [$\rm{Gpc}^{-3}\,M_{\odot}^{-1}$]')
ax3.set_xlabel(r'$M_{\mathrm{BH,1}}$ [$M_{\odot}$]')

ax4 = ax3.twinx()

# Plot data on the second y-axis
color = 'navy'
ax4.set_ylabel(r'$\mathrm{d}N_{\rm{detected}} / \mathrm{d}M_{\mathrm{BH,1}}$ $[M_{\odot}^{-1}$]')
for ii, dNdm1 in enumerate(dN_dm1_list):
    ax4.plot(mass_1, dNdm1, color=cs2[ii], label=labels_snr[ii])
ax4.set_ylim(1e-4, 5e2)
ax4.set_xlim(0, 95)
ax4.set_yscale('log')
ax4.legend(loc='lower right')
ax4.tick_params(axis='y', labelsize=12)
fig.tight_layout()


plt.savefig('fig1.png', dpi=150, facecolor='white')