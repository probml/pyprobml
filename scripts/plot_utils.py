import os
import matplotlib.pyplot as plt

DEFAULT_WIDTH = 6.0
# GOLDEN_MEAN = (5**0.5 - 1.0) / 2.0  # Aesthetic ratio
DEFAULT_HEIGHT = 1.5
SIZE_SMALL = 9  # Caption size in the book
DEFAULT_FIG_PATH = "figures"
# SPLINE_COLOR = 'gray'


def latexify(
    width_scale_factor=1,
    height_scale_factor=1,
    fig_width=None,
    fig_height=None,
    font_size=SIZE_SMALL,
):
    f"""
    width_scale_factor: float, DEFAULT_WIDTH will be divided by this number, DEFAULT_WIDTH is page width: {DEFAULT_WIDTH} inches.
    height_scale_factor: float, DEFAULT_HEIGHT will be divided by this number, DEFAULT_HEIGHT is {DEFAULT_HEIGHT} inches.
    fig_width: float, width of the figure in inches (if this is specified, width_scale_factor is ignored)
    fig_height: float, height of the figure in inches (if this is specified, height_scale_factor is ignored)
    font_size: float, font size
    """
    if fig_width is None:
        fig_width = DEFAULT_WIDTH / width_scale_factor
    if fig_height is None:
        fig_height = DEFAULT_HEIGHT / height_scale_factor

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams["pdf.fonttype"] = 42

    # Font sizes
    # SIZE_MEDIUM = 14
    # SIZE_LARGE = 24
    # https://stackoverflow.com/a/39566040
    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title

    # latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    plt.rcParams["backend"] = "ps"
    if not "NO_SAVE_FIGS" in os.environ:  # To remove latex dependency from GitHub actions
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(fig_width, fig_height))


def savefig(f_name, fig_dir=DEFAULT_FIG_PATH, tight_layout=True, *args, **kwargs):
    fname_full = os.path.join(fig_dir, f_name)

    if not "NO_SAVE_FIGS" in os.environ:
        print("saving image to {}".format(fname_full))
        if tight_layout:
            plt.tight_layout(pad=0)
        print("Figure size:", plt.gcf().get_size_inches())
        plt.savefig(fname_full, pad_inches=0.0, pad=0, h_pad=0, w_pad=0, *args, **kwargs)
        # bbox_inches="tight",  # This changes the size of the figure
