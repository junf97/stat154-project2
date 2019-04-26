import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import load_data, set_seed


def label_stats(df):
    """
    Show statistics about expert labels:
    - Total number of pixels in each image
    - Percentage of pixels in each class
    - Check if any two observations has the same coordinates in each image

    :param df: Data source (DataFrame)
    """
    print("> Number of pixels in each image:")
    print(df.groupby('source')['x'].count())
    print()

    print("> Percentage of pixels in each class:")
    print(pd.crosstab(df['source'], df['label'], normalize='index'))
    print()

    if df.groupby(['x', 'y', 'source']).size().max() == 1:
        print("> No observations with same coordinates found in each image")
    else:
        print("> Detected observations with same coordinates")


def feature_plot(df):
    """
    Plot overlay kde diagram for each feature within different images to
    compare distributions for the features.

    :param df: Data source (DataFrame)
    """
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.25)

    # All feature info (column name, feature name, subplot ax)
    features = [
        ('NDAI', 'NDAI', fig.add_subplot(2, 4, 1)),
        ('SD', 'SD', fig.add_subplot(2, 4, 2)),
        ('CORR', 'CORR', fig.add_subplot(2, 4, 3)),
        ('angle_DF', 'Radiance angle DF', fig.add_subplot(2, 4, 4)),
        ('angle_CF', 'Radiance angle CF', fig.add_subplot(2, 4, 5)),
        ('angle_BF', 'Radiance angle BF', fig.add_subplot(2, 4, 6)),
        ('angle_AF', 'Radiance angle AF', fig.add_subplot(2, 4, 7)),
        ('angle_AN', 'Radiance angle AN', fig.add_subplot(2, 4, 8))
    ]

    for label, df_source in df.groupby('source'):
        for feature, title, ax in features:
            ax.set_title(f'Distribution of {title}')
            df_source[feature].plot(kind="kde", ax=ax, label=label, legend=True)
            ax.set_xlabel(title)
    plt.show()


def label_plot(df):
    """
    Plot the distribution of expert labels based on x, y coordinates

    :param df: Data source (DataFrame)
    """
    fig = plt.figure(figsize=(13, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.21)
    # color = {1: '#B22222', 0: '#dddddd', -1: '#6495ED'}
    color = {1: '#dddddd', 0: 'black', -1: 'grey'}
    for idx, source in enumerate(df['source'].unique(), 1):
        ax = fig.add_subplot(1, 3, idx)
        ax.set_title(source)
        df_source = df.loc[df['source'] == source]
        df_source.plot.scatter(x='x', y='y', c=df_source['label'].apply(lambda x: color[x]), s=0.1, ax=ax)
    fig.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Cloud (label = 1)',
                               markerfacecolor=color[1], markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='No Cloud (label = -1)',
                               markerfacecolor=color[-1], markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Unlabeled (label = 0)',
                               markerfacecolor=color[0], markersize=10)
                        ], loc='lower center', bbox_to_anchor=(0.52, 0.01), ncol=3)
    plt.show()


if __name__ == '__main__':
    set_seed(0)

    data = load_data()
    label_stats(data)
    label_plot(data)
    feature_plot(data)
