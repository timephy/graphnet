from graphnet.pisa.utils import calculate_1D_contours, calculate_2D_contours

# This configuration dictionary overwrites our pisa standard with your preferences.
# note: num_bins should not be higer than 25 for reconstructions.
config_dict = {'reco_energy': {'num_bins': 10},
               'reco_coszen': {'num_bins': 10},
               'pid': {'bin_edges': [0, 0.30, 0.90, 1]},
               'true_energy': {'num_bins': 200},
               'true_coszen': {'num_bins': 200}}

outdir = '/home/iwsatlas1/timg/code/contour/pisa_out3'  # where you want the .csv-file with the results
run_name = 'test'    # what you call your run
pipeline_path = '/home/iwsatlas1/timg/code/contour/pipeline_results/pipeline_tim_0.db'

# Fit 1D contours
# calculate_1D_contours(outdir,
#                       run_name + '_1D',
#                       pipeline_path,
#                       model_name='dynedge',
#                       include_retro=True,
#                       config_dict=config_dict,
#                       grid_size=30,
#                       n_workers=1,
#                       statistical_fit=True)

# Fit 2D contours
calculate_2D_contours(outdir,
                      run_name + '_2D',
                      pipeline_path,
                      model_name='dynedge',
                      include_retro=True,
                      config_dict=config_dict,
                      grid_size=30,
                      n_workers=1,
                      statistical_fit=True)
