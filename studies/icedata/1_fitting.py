from graphnet.pisa.fitting import ContourFitter

# This configuration dictionary overwrites our pisa standard with your preferences.
# note: num_bins should not be higer than 25 for reconstructions.
config_dict = {
    "reco_energy": {"num_bins": 10},
    "reco_coszen": {"num_bins": 10},
    "pid": {"bin_edges": [0, 0.50, 1]},
    "true_energy": {"num_bins": 200},
    "true_coszen": {"num_bins": 200},
}

outdir = "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/fitting"  # where you want the .csv-file with the results
run_name = "run_1"  # what you call your run
pipeline_path = "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/idon_tilt-8#weighted-low#3/pipeline/pipeline.db"

fitter = ContourFitter(
    outdir=outdir,
    pipeline_path=pipeline_path,
    post_fix="_pred",
    model_name="dynedge",
    include_retro=True,
    statistical_fit=True,
)
# Fits 1D contours of dm31 and theta23 individually
fitter.fit_1d_contour(
    run_name=run_name + "_1d",
    config_dict=config_dict,
    grid_size=30,
    n_workers=30,
)
# Fits 2D contours of dm31 and theta23 together
fitter.fit_2d_contour(
    run_name=run_name + "_2d",
    config_dict=config_dict,
    grid_size=30,
    n_workers=30,
)
