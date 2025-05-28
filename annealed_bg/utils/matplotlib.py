import matplotlib.pyplot as plt
import scienceplots

import annealed_bg.utils.fessa


def setup_matplotlib_defaults():
    plt.set_cmap("fessa")
    plt.style.use(["science", "nature", "bright", "no-latex"])
