import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np

import os, sys
import cv2

from .playground import Map, Station, Bus

def aggregate_flow(flow, density):
    density = 3 * density
    t_all = len(flow)

    time = np.arange(max(t_all // density + 1, 1)) / (3 * 60) * density
    res = np.zeros_like(time)
    for t in range(t_all):
        res[t // density] += flow[t]
    width = density / (3 * 60)
    return time, res, width

def get_queue(s : Station):
    queue_array = np.zeros([len(s.queue), s.M.t + 1])
    for target_ind in range(s.M.n_station):
        for time, flow in s.queue[target_ind]:
            queue_array[target_ind][time] = flow
    return queue_array

def vis_flow(M : Map, density=5):
    n = M.n_station
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    spec = gridspec.GridSpec(ncols=n, nrows=n, figure=fig)
    for i, station in enumerate(M.stations):
        flows = get_queue(station)
        for j, flow in enumerate(flows):
            ax = fig.add_subplot(spec[i, j])
            time, res, width = aggregate_flow(flow, density)
            ax.bar(time, res, width)
            if i == n - 1:
                ax.set_xlabel(M.stations[j].name)
            if j == 0:
                ax.set_ylabel(M.stations[i].name)
    plt.savefig("flow.jpg")
    plt.close(fig)


def plot_station(fig, ax, station : Station, station_v_cat, C, scale, bar=True, scale_bar=10):
    i = station.ind
    node = station.node
    crd = node.crd
    station_v = [station_v_cat[:, 0], station_v_cat[:, 1:]]

    queue = station_v[0].cpu().numpy()

    scalex, scaley = scale
    font_size = scaley * 800

    ax.add_patch(plt.Circle(crd, radius=30, color=C[i]))
    ax.text(crd[0], crd[1] - 70, "{}".format(int(sum(queue))), ha='center', va='center', fontsize=font_size)
    
    if bar:
        bar_scale_x = scalex * 10
        bar_scale_y = scaley * 10
        
        crd_base_bar = [crd[0], crd[1] + 40]
        pix_crd_base_bar = ax.transData.transform(crd_base_bar)
        bar_x, bar_y = ax.transAxes.inverted().transform(pix_crd_base_bar)
        ax_bar = fig.add_axes([bar_x - bar_scale_x / 2, bar_y, bar_scale_x, bar_scale_y * 10])
        ax_bar.bar(np.arange(len(queue)), queue, color=C)
        ax_bar.set_ylim(0, scale_bar * 10)
        ax_bar.axis('off')

def plot_bus(fig, ax, bus : Bus, bus_vec, C, scale, stack_count, bar=True):

    w = 160
    h = 40
    active = bus_vec[0].cpu().item()
    loc = int(bus_vec[2].cpu().item())

    crd = [bus.M.nodes[loc].crd[0], bus.M.nodes[loc].crd[1]]
    passengers = bus_vec[4:].cpu().numpy()
    capacity = bus.capacity
    ind = bus.ind

    # stack the buses if there are more than 1 at the same place
    crd[1] += h * stack_count

    rect_anchor = [crd[0] - w/2, crd[1] - h/2]
    scalex, scaley = scale
    font_size = scaley * 800
    
    if bar:
        crd_base_bar = rect_anchor
        bar_x, bar_y = ax.transAxes.inverted().transform(ax.transData.transform(crd_base_bar))
        
        crd_rect_vert = [crd_base_bar[0] + w, crd_base_bar[1] + h]
        bar_x_vert, bar_y_vert = ax.transAxes.inverted().transform(ax.transData.transform(crd_rect_vert))
        
        bar_scale_x = bar_x_vert - bar_x
        bar_scale_y = bar_y_vert - bar_y

        ax_bar = fig.add_axes([bar_x, bar_y, bar_scale_x, bar_scale_y])
        total_count = 0
        for i, count in enumerate(passengers):
            ax_bar.barh([0], count, left=total_count, color=C[i])
            total_count += count
        ax_bar.set_ylim(0, 0.1)
        ax_bar.set_xlim(0, capacity)
        ax_bar.axis('off')
    
    bus_color = 'k' if active else 'b'
    ax.add_patch(plt.Rectangle(rect_anchor, width=w, height=h, ec=bus_color, fc='w'))
    ax.text(crd[0] + w/2 + 10, crd[1], "{}/{}".format(int(sum(passengers)), capacity), ha='left', va='center', fontsize=font_size, color=bus_color)
    ax.text(crd[0] - w/2 - 10, crd[1], "#{}".format(ind), ha='right', va='center', fontsize=font_size, color=bus_color)

def get_map_image(M, x_scale, y_scale, br=0.5):
    center_lat, center_long = M.center_lat, M.center_long * M.correction_long
    im = cv2.imread('./config/dukemap.png')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = (im * br + 255 * (1 - br)).astype(int)
    pix_scale = 1.5
    x_shift = 310
    y_shift = 230
    cropped_im = im[y_shift : y_shift + int(y_scale * pix_scale), x_shift : x_shift + int(x_scale * pix_scale)]
    return cropped_im

def vis_map(M : Map, vec, height=10, margin=0.3, cmap='jet', background=True):

    vec_bus, vec_station, t = vec

    width = height * (M.yspan * (1 + margin)) / (M.xspan + M.yspan * margin)
    fig = plt.figure(figsize=(height, width))
    cmap_func = cm.get_cmap(cmap)

    margin_scale = M.yspan * margin

    if background:
        ax_bg = fig.add_axes([1e-10, 0, 1, 1])
        background_img = get_map_image(M, M.xspan + margin_scale, M.yspan + margin_scale)
        ax_bg.imshow(background_img, zorder=-1, extent=[-M.xspan - margin_scale, M.xspan + margin_scale, -M.yspan - margin_scale, M.yspan + margin_scale])
        # ax_bg.set_xlim(0, 1000)
    
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_ylim(-M.yspan - margin_scale, M.yspan + margin_scale)
    ax.set_xlim(-M.xspan - margin_scale, M.xspan + margin_scale)
    # plot the routes
    scaley = height / M.yspan / (1 + margin)
    scalex = scaley / M.xspan * M.yspan
    scale = [scalex, scaley]

    wait_reward, opr_reward, queue_length, bus_efficiency, bus_efficiency_raw = M.reward_rule.rewards_terms(vec)
    ax.text(scalex * 5, scaley * 3, "t={}\nQueue:{}  WaitCost={:.4g}  OprCost={:.4g}  BusEfficiency={:.3g}%".format(t, int(queue_length), -wait_reward, -opr_reward, bus_efficiency_raw * 100), transform=ax.transAxes, fontsize=scaley * 1200)

    for edge in M.edges:
        i, j = edge
        ax.plot([M.nodes[i].crd[0], M.nodes[j].crd[0]], [M.nodes[i].crd[1], M.nodes[j].crd[1]], c='k', zorder=0, lw=0.5)


    C = [cmap_func(i) for i in np.linspace(0.2, 1, M.n_station)]
    for i, station in enumerate(M.stations):
        station_v = vec_station[i]
        plot_station(fig, ax, station, station_v, C, scale)
    
    bus_stack = np.zeros(len(M.nodes))
    for i, bus in enumerate(M.buses):
        loc = int(vec_bus[i][2].cpu().item())
        plot_bus(fig, ax, bus, vec_bus[i], C, scale, bus_stack[loc])
        bus_stack[loc] += 1

    ax.set_aspect('equal')
    ax.axis('off')
    # ax3 = fig.add_axes([0.99, 0.99, 0.1, 0.])
    # ax3.hist([-1, -1, 1, 1, 1, 1, 0])
    plt.savefig('sample.png', dpi=150)
    plt.close(fig)
