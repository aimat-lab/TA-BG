# Fessa palette for python matplotlib
# Source: https://github.com/luigibonati/fessa-color-palette/blob/master/fessa.py

import matplotlib
from matplotlib.colors import ColorConverter, LinearSegmentedColormap

paletteFessa = [
    "#1F3B73",  # dark-blue
    "#2F9294",  # green-blue
    "#50B28D",  # green
    "#A7D655",  # pisello
    "#FFE03E",  # yellow
    "#FFA955",  # orange
    "#D6573B",  # red
]

cm_fessa = LinearSegmentedColormap.from_list("fessa", paletteFessa)
matplotlib.colormaps.register(cmap=cm_fessa)
matplotlib.colormaps.register(cmap=cm_fessa.reversed())

for i in range(len(paletteFessa)):
    ColorConverter.colors[f"fessa{i}"] = paletteFessa[i]

### To set it as default
# import fessa
# plt.set_cmap('fessa')
### or the reversed one
# plt.set_cmap('fessa_r')
### For contour plots
# plt.contourf(X, Y, Z, cmap='fessa')
### For standard plots
# plt.plot(x, y, color='fessa0')
