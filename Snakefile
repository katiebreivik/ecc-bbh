#rule simulategrid:
#    input:
#        "src/data/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5"
#    output:
#        "src/data/t_LIGO.npy",
#        "src/data/e_LISA.npy",
#        "src/data/comoving_volume.npy",
#        "src/data/horizon_distance.npy"
#    script:
#        "src/scripts/simulategrid.py"
        
rule fig1:
    input:
        "src/data/low_res/t_LIGO.npy",
        "src/data/low_res/e_LISA.npy",
        "src/data/low_res/comoving_volume.npy",
        "src/data/low_res/horizon_dist.npy"
    output:
        "src/tex/figures/fig1.pdf"
    script:
        "src/scripts/fig1.py"

rule fig2:
    input:
        "src/data/low_res/t_LIGO.npy",
        "src/data/low_res/e_LISA.npy",
        "src/data/low_res/comoving_volume.npy",
        "src/data/low_res/horizon_dist.npy"
    output:
        "src/tex/figures/fig2.pdf"
    script:
        "src/scripts/fig2.py"