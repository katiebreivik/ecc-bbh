import legwork as lw
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18, z_at_value
import numpy as np
from scipy.interpolate import interp1d


def get_LIGO_rate(down_samp_fac_m1=1, down_samp_fac_q=1):
    ''' loads in LVK Power Law + Peak Model data and returns
    the rate on a grid that may be downsampled

    Parameters
    ----------
    down_samp_fac_m1 : `int`
        factor of downsample to take for grid that is 
        originally of size: 500000
    
    down_samp_fac_q : `int`
        factor of downsample to take for grid that is 
        originally of size: 500000

    Returns
    -------
    mass_1 : `numpy.array`
        mass 1 grid
    
    mass_ratio`numpy.array`
        mass ratio grid
    
    M1 : `numpy.array`
        mass 1 meshgrid
    
    Q : `numpy.array`
        mass ratio meshgrid
    
    dN_dm1dqdVcdt : `numpy.array`
        BBH merger rate per unit m1, q, comoving volume on meshgrid
    '''

    import deepdish as dd
     # this is lifted ~exactly~ from the GWTC-3 tutorial
    mass_PP_path = paths.data / "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5"
    with open(mass_PP_path, 'r') as _data:
        _data = dd.io.load(mass_PP_path)
    
    dN_dm1dqdVcdt = _data['ppd']
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    M1, Q = np.meshgrid(mass_1, mass_ratio, indexing='ij')
    
    if down_samp_fac_m1 > 1:
        mass_1 = mass_1[::down_samp_fac_m1]
        mass_ratio = mass_ratio[::down_samp_fac_q]
        M1 = M1[::down_samp_fac_m1, ::down_samp_fac_q]
        Q = Q[::down_samp_fac_m1, ::down_samp_fac_q]
        dN_dm1dqdVcdt = dN_dm1dqdVcdt[::down_samp_fac_q, ::down_samp_fac_m1]

    return mass_1*u.Msun, mass_ratio, M1*u.Msun, Q, dN_dm1dqdVcdt*u.Msun**(-1) * u.Gpc**(-3) * u.yr**(-1)


def get_e_LIGO(e_LISA, f_LISA, m1, m2):
    e_grid_steps = 5000
    a_LISA = lw.utils.get_a_from_f_orb(f_orb=f_LISA, m_1=m1, m_2=m2)
    e_evol_grid = np.logspace(np.log10(1e-15), np.log10(e_LISA), e_grid_steps)
    a_evol_grid = lw.utils.get_a_from_ecc(e_evol_grid, lw.utils.c_0(a_i=a_LISA, ecc_i=e_LISA))
    log_a_interp = interp1d(a_evol_grid.to(u.Rsun).value, e_evol_grid)
    a_LIGO = lw.utils.get_a_from_f_orb(f_orb=10 * u.Hz, m_1=m1, m_2=m2) 
    e_LIGO = log_a_interp(a_LIGO.to(u.Rsun).value)

    return e_LIGO

def get_e_LISA(e_LIGO, f_LISA, m1, m2):
    if e_LIGO <= 1e-6:
        e_grid_steps = 500
    elif e_LIGO < 1e-5:
        e_grid_steps = 1000
    else:
        e_grid_steps = 5000
    a_LIGO = lw.utils.get_a_from_f_orb(f_orb=10 * u.Hz, m_1=m1, m_2=m2) 
    a_LISA = lw.utils.get_a_from_f_orb(f_orb=f_LISA, m_1=m1, m_2=m2)
    e_evol_grid = np.logspace(np.log10(e_LIGO), 0.999, e_grid_steps)
    a_evol_grid = lw.utils.get_a_from_ecc(e_evol_grid, lw.utils.c_0(a_i=a_LIGO, ecc_i=e_LIGO))
    log_a_interp = interp1d(a_evol_grid.to(u.Rsun).value, e_evol_grid)
    e_LISA = log_a_interp(a_LISA.to(u.Rsun).value)
    return float(e_LISA)

def get_T_LIGO(e_LISA, f_LISA, m1, m2):
    beta = lw.utils.beta(m1, m2)
    a_LISA = lw.utils.get_a_from_f_orb(f_orb=f_LISA, m_1=m1, m_2=m2)
    Tc = a_LISA**4 / (4 * beta)

    ecc_fac = (1 - e_LISA**2)**(7/2) * (1 + 0.27 * e_LISA**10 + 0.33 * e_LISA**20 + 0.2 * e_LISA**1000)
    
    return Tc * ecc_fac


def get_e_LIGO_t_LIGO(dat_in):
    e, f, m1, m2 = dat_in
    e_LIGO = get_e_LIGO(e, f, m1, m2)
    T_LIGO = get_T_LIGO(e, f, m1, m2).to(u.yr).value
    
    return [e_LIGO, T_LIGO]

def get_e_LISA_t_LIGO(dat_in):
    e, f, m1, m2 = dat_in
    if e > 0:
        e_LISA = get_e_LISA(e, f, m1, m2)
        T_LIGO = get_T_LIGO(e, f, m1, m2).to(u.yr).value
    else:
        e_LISA = 0
        beta = lw.utils.beta(m1, m2)
        a_LISA = lw.utils.get_a_from_f_orb(f_orb=f, m_1=m1, m_2=m2)
        T_LIGO = (a_LISA**4 / (4 * beta)).to(u.yr).value
    
    return e_LISA, T_LIGO

def get_horizon(m1, m2, e, f_orb, snr_thresh=12):
    source = lw.source.Source(m_1=m1,
                              m_2=m2,
                              ecc=e,
                              f_orb=f_orb,
                              dist=8 * u.Mpc,
                              interpolate_g=False,
                              gw_lum_tol=0.05)

    snr = source.get_snr(approximate_R=True, verbose=False, t_obs=8*u.yr)

    d_horizon = snr / snr_thresh * 8 * u.Mpc

    return d_horizon


def get_Vc_Dh(dat_in):
    m1, q, e, f, snr_thresh = dat_in
    m_1 = m1
    m_2 = m1 * q
    d_h = get_horizon(m1=m_1, m2=m_2, e=e, f_orb=f, snr_thresh=snr_thresh)
    #Calculate the comoving volume based on the horizon distance + cosmology if necessary
    if d_h > 10*u.kpc:
        redshift = z_at_value(Planck18.luminosity_distance, d_h)
        V_c = Planck18.comoving_volume(z=redshift)
    else:
        V_c = 4/3 * np.pi * d_h**3  

    return [d_h.to(u.Gpc).value, V_c.to(u.Gpc**3).value]


def chunk_list(long_list, num_chunks):
    avg = len(long_list) / float(num_chunks)
    chunks = []
    last = 0.0

    while last < len(long_list):
        chunks.append(long_list[int(last):int(last + avg)])
        last += avg

    return chunks

def get_VC_new(snr_thresh_new, DH, VC):
    snr_thresh_data = 12
    DH = DH * snr_thresh_data / snr_thresh_new
    VC = 4/3 * np.pi * DH**3

    return VC