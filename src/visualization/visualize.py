def annot_plot(ax, w, h):
    """Add annotations to bar plot

    From https://www.kaggle.com/pankajjsh06/ipl-data-2008-2018
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate(
            '{0:.1f}'.format(p.get_height()), (p.get_x() + w, p.get_height() + h)
        )
