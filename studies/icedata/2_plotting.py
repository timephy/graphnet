from graphnet.pisa.plotting import plot_2D_contour, plot_1D_contour

# 2D CONTOUR PLOT EXAMPLE

# Here we would like to plot two contours in one figure; one for our GNN and one for retro.
# We build a dictionary  for each contour. Each dictionary much contain "path" and "model".
# "path" is the path to the .csv file containing the fit result.
# "model" is the name of the model in the .csv file - some fits have more than 1 model!
# The plotting script returns the figure object - remember to save it yourself!

contour_data_2D = []
# contour_data_2D.append(
#     {
#         "path": "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/fitting/merged_results.csv",
#         "model": "dynedge",
#         "label": "dynedge",
#         "color": "tab:blue",
#     }
# )
contour_data_2D.append(
    {
        "path": "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/fitting/merged_results.csv",
        "model": "retro",
        "label": "retro",
        "color": "tab:orange",
    }
)

figure = plot_2D_contour(contour_data_2D, width=6.3, height=2.3 * 2)
figure.savefig(
    "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/2_plotting/contour_rasmus.pdf"
)

exit()

contour_data_2D = []
contour_data_2D.append(
    {
        "path": "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/fitting/run_1_2d/merged_results.csv",
        "model": "dynedge",
        "label": "dynedge",
        "color": "tab:blue",
    }
)
contour_data_2D.append(
    {
        "path": "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/fitting/run_1_2d/merged_results.csv",
        "model": "retro",
        "label": "retro",
        "color": "tab:orange",
    }
)

figure = plot_2D_contour(contour_data_2D, width=6.3, height=2.3 * 2)
figure.savefig(
    "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/2_plotting/2d_contour_test.pdf"
)

# 1D CONTOUR PLOT EXAMPLE
contour_data_1D = []
contour_data_1D.append(
    {
        "path": "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/fitting/run_1_1d/merged_results.csv",
        "color": "tab:orange",
        "model": "retro",
        "label": "retro - vanilla bin",
        "ls": "--",
    }
)
contour_data_1D.append(
    {
        "path": "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/fitting/run_1_1d/merged_results.csv",
        "color": "tab:blue",
        "model": "dynedge",
        "label": "dynedge - vanilla bin",
        "ls": "--",
    }
)
figure = plot_1D_contour(contour_data_1D)
figure.savefig(
    "/home/iwsatlas1/timg/code/graphnet/studies/icedata/archive/2_plotting/1d_contour_test.pdf"
)
