import copy
import logging
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FONTSIZE = 45
MARKSIZE = 200
COLOR_PLAT = ['#F38181', '#FCE38A', '#EAFFD0', '#95E1D3']


def draw_interation(cube1='Autotune Center',
                    cube2='FS Runner',
                    arrow_dir='down',
                    arrow_len=6,
                    info=None,
                    max_width=20,
                    font_size=50):
    def wrap_text(text, text_width=20):
        if len(text) > text_width - len('...'):
            return textwrap.shorten(text, width=text_width, placeholder="...")
        else:
            return ' ' * (text_width - len(text)) + text + ' ' * (text_width -
                                                                  len(text))

    x, y = 0.25, 0.4  # Initial position
    size, max_len = font_size, 10
    width, height = (size / max_len) * 0.07, 0.5

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot()

    # Cubes
    ax.text(x,
            y + height,
            wrap_text(cube1, max_width),
            size=size,
            bbox=dict(boxstyle="round4", alpha=0.8, color='#85CDFD'))

    ax.text(x,
            y,
            wrap_text(cube2, max_width),
            size=size,
            bbox=dict(boxstyle="round4", alpha=0.8, color='#85CDFD'))

    # Arrows
    a_x = x + width * 0.75
    a_y = (y + height) / 2 + 0.1
    # Down
    ax.text(x + width * 0.3,
            a_y,
            wrap_text(' ', arrow_len),
            size=size,
            rotation=270,
            bbox=dict(boxstyle="rarrow",
                      color='green' if arrow_dir == 'down' else 'grey'))
    # Up
    ax.text(a_x,
            a_y,
            wrap_text(' ', arrow_len),
            size=size,
            rotation=90,
            bbox=dict(boxstyle="rarrow",
                      color='grey' if arrow_dir == 'down' else 'green'))

    if info:
        # Bubbles
        ix, iy = a_x + 0.08, a_y
        for i in range(3):
            ax.text(ix,
                    iy,
                    ' ',
                    size=20 + i * 3,
                    rotation=270,
                    bbox=dict(boxstyle="circle", alpha=0.4, color='#F2DEBA'))
            ix, iy = ix + 0.03, iy + 0.03
        # Information
        info = textwrap.fill(info,
                             width=max_width,
                             max_lines=6,
                             placeholder="...")
        plt.text(ix,
                 iy,
                 info,
                 size=40,
                 bbox=dict(
                     boxstyle="sawtooth",
                     facecolor='#F2DEBA',
                     edgecolor='black',
                 ))
    plt.axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig


def draw_landscape(df, diagnosis_configs, larger_better, metric):
    import seaborn as sns

    df = copy.deepcopy(df)
    landscape_1d = {}
    col_name = df.columns

    # 1D landscape
    for hyperparam in diagnosis_configs:
        if hyperparam not in col_name:
            logger.warning(f'Invalid hyperparam name: {hyperparam}')
            continue
        else:
            df[hyperparam] = df[hyperparam].apply(str)
            plt.figure(figsize=(20, 15))
            ranks = list(
                df.groupby(hyperparam)["performance"].mean().fillna(
                    0).sort_values()[::-1].index)
            # Set color
            palette = {
                x: COLOR_PLAT[c % len(COLOR_PLAT)]
                for c, x in enumerate(ranks)
            }
            if not larger_better:
                ranks.reverse()
            sns.boxplot(
                x="performance",
                y=hyperparam,
                data=df,
                order=ranks,
                width=.2,
                # saturation=0.3,
                notch=True,
                palette=palette)
            sns.stripplot(x="performance",
                          y=hyperparam,
                          data=df,
                          jitter=True,
                          color="black",
                          size=10,
                          linewidth=0,
                          order=ranks)
            plt.yticks(rotation=45, fontsize=FONTSIZE)
            plt.xticks(rotation=10, fontsize=FONTSIZE)
            plt.xlabel(metric, size=FONTSIZE)
            plt.ylabel("Higher the better", size=FONTSIZE)
            plt.title(f"Rank of {hyperparam}", fontsize=FONTSIZE)
            sns.despine(trim=True)
            plt.tight_layout()
            landscape_1d[f"{hyperparam}"] = plt.gcf()
            plt.close()
    return landscape_1d


def draw_pca(df, metric='loss'):
    import seaborn as sns
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    X = df.iloc[:, :-1]

    for col in X.columns.tolist():
        X[col] = X[col].astype('category')
        X[col] = X[col].cat.codes
    X_std = preprocessing.scale(X)
    pca = PCA(n_components=1)
    pca.fit(X_std)
    X_pca = pd.DataFrame(
        pca.fit_transform(X_std)).rename(columns={0: 'component'})
    Y = pd.DataFrame(df["performance"])
    data_pca = pd.concat([X_pca, Y], axis=1)

    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=10,
                                   alpha=0.1)
    reg.fit([[x] for x in data_pca['component'].tolist()],
            data_pca['performance'].tolist())
    x_ticks = np.linspace(np.min(data_pca['component']),
                          np.max(data_pca['component']), 100)
    ys = reg.predict([[x] for x in x_ticks])

    plt.figure(figsize=(20, 15))
    sns.scatterplot(data=data_pca, x='component', y='performance', s=MARKSIZE)
    gp = pd.DataFrame(dict(x=x_ticks, y=ys))
    sns.lineplot(data=gp, x='x', y='y')

    plt.title("Gaussian", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel("ConfigSpace", size=FONTSIZE)
    plt.ylabel("metric", size=FONTSIZE)
    plt.tight_layout()
    pca = plt.gcf()
    plt.close()
    return pca


def draw_info(trial, config, metric):
    plt.figure(figsize=(30, 15))
    anc_x, anc_y, bias, dy = 0.5, 0.8, 0, 0.14
    fontsize = 75
    texts = [
        f"Trial [{trial}] ongoing:",
        config,
        f"Monitoring `{metric}`...",
    ]

    for line, t in enumerate(texts):
        if isinstance(t, str):
            plt.text(anc_x,
                     anc_y + bias,
                     t,
                     size=fontsize,
                     ha="center",
                     va="center",
                     bbox=dict(
                         boxstyle="round4" if line == len(texts) -
                         1 else 'sawtooth',
                         facecolor='lightblue',
                         edgecolor='black',
                     ))
            bias -= dy
        elif isinstance(t, dict):
            for key, value in t.items():
                plt.text(anc_x,
                         anc_y + bias,
                         f"{key}: {value}",
                         size=fontsize,
                         ha='center',
                         va="center",
                         bbox=dict(
                             boxstyle="sawtooth",
                             facecolor='none',
                             edgecolor='black',
                         ))
                bias -= dy

    plt.axis('off')
    plt.tight_layout()
    info = plt.gcf()
    plt.close()
    return info


def draw_para_coo(df):
    import plotly.graph_objects as go

    new_df = copy.deepcopy(df)
    px_layout = []
    for col in new_df.columns.tolist():
        if isinstance(new_df[col][0], str):
            new_df[col] = new_df[col].astype('category')
            cat_map = dict(zip(new_df[col].cat.codes, new_df[col]))
            px_layout.append({
                'range': [min(cat_map.keys()),
                          max(cat_map.keys())],
                'label': col,
                'tickvals': list(cat_map.keys()),
                'ticktext': list(cat_map.values()),
                'values': new_df[col].cat.codes,
            })
        else:
            px_layout.append({
                'range': [0, np.nanmax(new_df[col])],
                'label': col,
                'values': new_df[col],
            })
    new_df['Trial Index'] = range(1, len(new_df) + 1)

    px_fig = go.Figure(data=go.Parcoords(line=dict(
        color=new_df['Trial Index'],
        colorscale='YlOrRd',
        showscale=True,
        cmin=1,
        cmax=len(new_df) + 1,
        colorbar=dict(dtick=len(new_df) // 2, tickprefix='Trial')),
                                         dimensions=px_layout,
                                         labelangle=18,
                                         labelside='bottom'))
    px_fig.update_layout(font={'size': 18})
    return px_fig.to_image(format="png")
