import legwork as lw
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18, z_at_value
import numpy as np
from scipy.interpolate import interp1d
import paths


def get_a_from_f(f_orb, m1, m2):
    ''' gets the semi-major axis from the frequency using Kepler's third law
    requires astropy units

    Parameters
    ----------
    f_orb : `float`
        frequency in Hz
    m1 : `float`
        mass 1 in solar masses
    m2 : `float`
        mass 2 in solar masses

    Returns
    -------
    a : `float`
        semi-major axis in AU
    '''
    
    G = 6.67e-11
    Msun = 1.989e30 # kilograms
    Rsun = 6.96e8 # meters
    P = 1 / f_orb # period in seconds
    # Convert to semi-major axis in meters
    a = (G * (m1*Msun + m2*Msun) * P**2 / (4 * np.pi**2))**(1/3)
    a = a / Rsun # convert to Rsun
    return a

def get_beta(m1, m2):
    ''' gets the beta parameter from the masses using Peters and Mathews (1964) Eq.5.11
    requires astropy units

    Parameters
    ----------
    m1 : `float`
        mass 1 in solar masses
    m2 : `float`
        mass 2 in solar masses

    Returns
    -------
    beta : `float`
        beta parameter
    '''
    G = 6.67e-11 # gravitational constant in m^3 kg^-1 s^-2
    c = 3e8 # speed of light in m/s
    Msun = 1.989e30 # mass of the sun in kg
    # Convert masses to kg
    m1 = m1 * Msun
    m2 = m2 * Msun
    # Calculate the beta parameter
    beta = (G * (m1 + m2) / c**3)**(5/3)
    return beta


def get_c_0(a_i, ecc_i):
    ''' gets the initial semi-major axis from the initial eccentricity
    using Kepler's third law; requires astropy units

    Parameters
    ----------
    a_i : `float/array`
        Initial semi-major axis

    ecc_i : `float/array`
        Initial eccentricity

    Returns
    -------
    c_0 : `float`
        Constant defined in Peters and Mathews (1964) Eq.5.11
    '''
    
    c_0 = a_i * (1 - ecc_i**2) * ecc_i**(-12/19) * (1 + (121/304)*ecc_i**2)**(-870/2299)
    return c_0

def get_a_from_ecc(ecc, c_0):
    '''Convert eccentricity to semi-major axis

    Use initial conditions and Peters (1964) Eq. 5.11 to convert ``ecc`` to ``a``.

    Parameters
    ----------
    ecc : `float/array`
        Eccentricity

    c_0 : `float`
        Constant defined in Peters and Mathews (1964) Eq. 5.11. See :meth:`legwork.utils.c_0`

    Returns
    -------
    a : `float/array`
        Semi-major axis'''

    a = c_0 * ecc**(12/19) / (1 - ecc**2) * (1 + (121/304) * ecc**2)**(870/2299)
    
    return a


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

    # no units since they wreck the vector manipulation below
    return mass_1, mass_ratio, M1, Q, dN_dm1dqdVcdt



def dTmerger_df(m1, m2, f, e):
    ecc_fac = (1 - e**2)**(7/2) * (1 + 0.27 * e**10 + 0.33 * e**20 + 0.2 * e**1000)
    f_dot = lw.utils.fn_dot(m_c=lw.utils.chirp_mass(m1, m2), e=np.zeros(f.shape), n=1, f_orb=f)
    
    dt_df = -5 * np.pi / 48 * c.c**5/(c.G * lw.utils.chirp_mass(m1, m2))**(5/3) * (2 * np.pi * f)**(-11/3)
    
    return ecc_fac / f_dot

def get_e_LIGO(e_LISA, f_LISA, m1, m2):
    e_grid_steps = 5000
    a_LISA = get_a_from_f(f_orb=f_LISA, m1=m1, m2=m2)
    e_evol_grid = np.logspace(np.log10(1e-15), np.log10(e_LISA), e_grid_steps)
    a_evol_grid = lw.utils.get_a_from_ecc(e_evol_grid, lw.utils.c_0(a_i=a_LISA, ecc_i=e_LISA))
    log_a_interp = interp1d(np.log10(a_evol_grid.to(u.Rsun).value), np.log10(e_evol_grid))
    a_LIGO = get_a_from_f(f_orb=10, m1=m1, m2=m2) 
    e_LIGO = 10**log_a_interp(np.log10(a_LIGO))

    return e_LIGO




def get_e_LISA(dat_in):
    ''' gets the eccentricity at LISA frequency given the 
    LIGO eccentricity by evaluating the Peters 1964 equation
    over a grid of eccentricities and interpolating; requires
    astropy units
    
    Parameters
    ----------
    e_LIGO : `float`
        eccentricity at LIGO frequency [f=10 Hz]
    f_LISA : `float`
        frequency at LISA [f in (10^-4 Hz, 10^-2 Hz)]
    m1 : `float`
        mass 1 in solar masses 
    m2 : `float`
        mass 2 in solar masses
    
    Returns
    -------
    e_LISA : `float`
        eccentricity at LISA frequency
    '''
    e_LIGO, f_LISA, m1, m2 = dat_in
    if e_LIGO <= 1e-6:
        e_grid_steps = 500
    elif e_LIGO < 1e-5:
        e_grid_steps = 1000
    else:
        e_grid_steps = 5000
    a_LIGO = lw.utils.get_a_from_f_orb(f_orb=10 * u.Hz, m_1=m1, m_2=m2) 
    a_LISA = lw.utils.get_a_from_f_orb(f_orb=f_LISA, m_1=m1, m_2=m2)
    e_evol_grid = np.logspace(np.log10(e_LIGO), np.log10(0.9999), e_grid_steps)
    a_evol_grid = lw.utils.get_a_from_ecc(e_evol_grid, lw.utils.c_0(a_i=a_LIGO, ecc_i=e_LIGO))
    
    log_a_interp = interp1d(np.log10(a_evol_grid.to(u.Rsun).value), np.log10(e_evol_grid))
    e_LISA = 10**log_a_interp(np.log10(a_LISA.to(u.Rsun).value))
    return float(e_LISA)

def get_T_LIGO(e_LISA, f_LISA, m1, m2):
    ''' gets the time to evolve between the LISA and LIGO frequencies
    using the Mandel Fit to eccentric evolution; requires astropy units

    Parameters
    ----------
    e_LISA : `float`
        eccentricity at LISA frequency
    f_LISA : `float`
        frequency at LISA [f in (10^-4 Hz, 10^-2 Hz)]
    m1 : `float`
        mass 1 in solar masses
    m2 : `float`
        mass 2 in solar masses
    Returns
    -------
    T_LIGO : `float`
        time to evolve between the LISA and LIGO frequencies in [yr]'''
    
    Rsun = 6.96e8 # meters
    yr = 3.154e7 # seconds
    # this is in SI units
    beta = get_beta(m1, m2)

    # this is in Rsun
    a_LISA = get_a_from_f(f_orb=f_LISA, m1=m1, m2=m2)
    
    Tc = (a_LISA*Rsun)**4 / (4 * beta) / yr

    ecc_fac = (1 - e_LISA**2)**(7/2) * (1 + 0.27 * e_LISA**10 + 0.33 * e_LISA**20 + 0.2 * e_LISA**1000)
    
    return Tc * ecc_fac


def get_e_LIGO_t_LIGO(dat_in):
    e, f, m1, m2 = dat_in
    e_LIGO = get_e_LIGO(e, f, m1, m2)
    T_LIGO = get_T_LIGO(e, f, m1, m2).to(u.yr).value
    
    return [e_LIGO, T_LIGO]


def get_e_LISA_t_LIGO(dat_in):
    ''' gets the eccentricity at LISA frequency then 
    calculates the time to evolve between the LISA and LIGO frequencies
    
    Parameters
    ----------
    dat_in : `list`
        list of [eccentricity, frequency, mass 1, mass 2]
        
    Returns
    -------
    e_LISA : `float`
        eccentricity at LISA frequency
    T_LIGO : `float`
        time to evolve between LISA and LIGO frequencies in [yr]
    '''
    e, f, m1, m2 = dat_in
    Rsun = 6.96e8 # meters
    yr = 3.154e7 # seconds

    circ_mask = e == 0

    e_LISA = np.zeros_like(e)
    T_LIGO = np.zeros_like(e)

    # Calculate e_LISA and T_LIGO for circular orbits
    e_LISA[circ_mask] = 0
    # this is in SI units
    beta = get_beta(m1[circ_mask], m2[circ_mask])

    # this is in Rsun
    a_LISA = get_a_from_f(f_orb=f[circ_mask], m1=m1[circ_mask], m2=m2[circ_mask])
    
    # this is in yr
    T_LIGO[circ_mask] = ((a_LISA*Rsun)**4 / (4 * beta))/yr

    # Calculate e_LISA and T_LIGO for eccentric orbits
    e_LISA[~circ_mask] = get_e_LISA(e[~circ_mask], f[~circ_mask], m1[~circ_mask], m2[~circ_mask])
    T_LIGO[~circ_mask] = get_T_LIGO(e[~circ_mask], f[~circ_mask], m1[~circ_mask], m2[~circ_mask])
    
    return e_LISA, T_LIGO


def get_horizon(m1, m2, e, f_orb, snr_thresh=12):
    ''' gets the horizon distance for a given set of parameters
    using the legwork source class; requires astropy units
    
    Parameters
    ----------
    m1 : `float`
        mass 1 in solar masses
    m2 : `float`
        mass 2 in solar masses
    e : `float`
        eccentricity
    f_orb : `float`
        frequency in Hz
    snr_thresh : `float`
        signal to noise ratio threshold; default is 12
    
    Returns
    -------
    d_horizon : `float`
        horizon distance in Mpc'''
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

    return d_h.to(u.Gpc).value, V_c.to(u.Gpc**3).value


def chunk_list(long_list, num_chunks):
    ''' splits a long list into smaller chunks for parallel processing

    Parameters
    ----------
    long_list : `list`
        list to be split into smaller chunks
    num_chunks : `int`
        number of chunks to split the list into

    Returns
    -------
    chunks : `list`
        list of smaller lists
    '''
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