rule simulategrid
    input:
        "src/data/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5"
    output:
        "src/data/t_LIGO.npy"
        "src/data/e_LISA.npy"
        "src/data/comoving_volume.npy"
        "src/data/horizon_distance.npy"
    script:
        "src/scripts/simulategrid.py"