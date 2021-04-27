import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib as mpl
from matplotlib import cm, rc
from os.path import join
import matplotlib.gridspec as gridspec
import os

mpl.use('Agg')
plt.style.use('ggplot')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
params= {'text.latex.preamble' : [r'\usepackage{amsmath}',r'\usepackage{amssymb}', r'\usepackage{bm}']}
plt.rcParams.update(params)

cmap_func = cm.get_cmap('Set1')

log_dirs = ['./model_log/3s_oneflow/2021-04-27-04-05-52-855265',
            './model_log/3s_oneflow/2021-04-27-04-15-16-267689',
            './model_log/3s_oneflow/2021-04-27-04-15-30-009055']

names = ['Random Agent', r'BDQ $\epsilon=0.2$', r'BDQ $\epsilon=0.1$']
plt_name = '3s_oneflow'

def get_log(log_dir):
    res = None
    json_fl = join(log_dir, 'stats.json')
    with open(json_fl) as infl:
        res = json.load(infl)
    return res

def get_image_path(name, store_format='png'):
    vis_dir = "/usr/xtmp/CSPlus/VOLDNN/Shared/Visualizations/"
    img_path = os.path.join(vis_dir, "{}.{}".format(name, store_format))
    print("https://users.cs.duke.edu/~xz231" + img_path.split("Shared")[1] + '\n')
    return img_path

def seq_moving_average(x, w):
    base = np.convolve(np.ones_like(x), np.ones(w), 'same')
    return np.convolve(np.array(x), np.ones(w), 'same') / base

def plot_result(ax, logs, names, key, title):
    for i, name in enumerate(names):
        raw_val = np.array(logs[i][key])
        smooth_val = seq_moving_average(raw_val, 20)
        tmp_color = cmap_func(i/5)
        x_base = np.arange(len(raw_val))
        ax.plot(x_base, raw_val, color=tmp_color, alpha=0.3)
        ax.plot(x_base, smooth_val, color=tmp_color, alpha=1, label=name)
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.legend()

def plot_log(log_dirs, names, plt_name):
    logs = [get_log(dirc) for dirc in log_dirs]
    plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3)
    plt.subplot(gs[0,0])
    plot_result(plt.gca(), logs, names, 'reward', 'Reward')
    plt.subplot(gs[0,1])
    plot_result(plt.gca(), logs, names, 'queue', 'Average Queue Length')
    plt.subplot(gs[0,2])
    plot_result(plt.gca(), logs, names, 'invalid_args', 'Invalid Args')
    plt.tight_layout()

    img_path = get_image_path(plt_name, store_format='pdf')
    plt.savefig(img_path, dpi=200, bbox='tight')

plot_log(log_dirs, names, plt_name)