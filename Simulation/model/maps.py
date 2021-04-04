import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import csv

from .utils import powerset

class MapNode():
    def __init__(self, ind, name, crd, is_station=True):
        self.ind = ind
        self.name = name
        self.crd = crd
        self.is_station = is_station
        self.neighbors = []

    def add_neighbor(self, neighbor):
        for n in self.neighbors:
            if n.ind == neighbor.ind:
                raise AssertionError
        self.neighbors.append(neighbor)
    
    def remove_neighbor(self, neighbor):
        for n in self.neighbors:
            if n.ind == neighbor.ind:
                self.neighbors.remove(n)
                return
    
    def __str__(self):
        return "Node {} {}".format(self.ind, self.name)
    
    def print_neighbors(self):
        print([str(x) for x in self.neighbors])
            

def all_paths_utils(u, e, visited, path, path_record):
    # Mark the current node as visited and store in path
    visited[u.ind] = True
    path.append(u.ind)
    if u.ind == e.ind:
        path_record.append(deepcopy(path))
    else:
        for neighbor in u.neighbors:
            if not visited[neighbor.ind]:
                all_paths_utils(neighbor, e, visited, path, path_record)
    path.pop()
    visited[u.ind] = False


def gen_routes(paths, station_inds, vertex_inds, power=False):
    all_routes = {}
    for path in paths:
        passing_stations = []
        for i in path[1: -1]:
            if i in vertex_inds:
                invalid = True
            if i in station_inds:
                passing_stations.append(i)

        psets = powerset(passing_stations)
        if not power:
            psets = [psets[-1]]
        
        for pset in psets:
            pset = [path[0]] + pset + [path[-1]]
            pset_str = "_".join([str(x) for x in pset])
            if pset_str not in all_routes:
                all_routes[pset_str] = [path, pset]
            else:
                if len(path) < len(all_routes[pset_str][0]):
                    all_routes[pset_str] = [path, pset]
    return all_routes

class DukeMap():
    def __init__(self, csv_fl, skip_station=False):
        self.nodes = []
        self.vertex_inds = [] # The collection of vertices (where bus can stop and change direction)
        self.station_inds = []
        self.routes = []
        self.route_names = []
        self.routes_dict = {}
        self.routes_choice = {}
        self.parse_csv(csv_fl)
        self.discreatize()
        self.get_paths(power=skip_station)
        return
    
    def parse_csv(self, csv_fl):
        with open(csv_fl, newline='') as csvfile:
            map_data = [x for x in csv.reader(csvfile)]
        vertices_names = map_data[-1]
        map_data = map_data[:-1]

        crds_long, crds_lat = [], []
        for i, node_info in enumerate(map_data):
            crds_long.append(float(node_info[3]))
            crds_lat.append(float(node_info[2]))
        center_long = (max(crds_long) + min(crds_long)) / 2
        center_lat = (max(crds_lat) + min(crds_lat)) / 2
        correction_long = np.cos(center_lat / 180 * np.pi)
        circ = 1.0015e7 / 90.0

        for i, node_info in enumerate(map_data):
            name = node_info[1]
            crd = np.array([(float(node_info[3]) - center_long) * circ * correction_long, (float(node_info[2]) - center_lat) * circ])
            is_station = name[0] != 'n'
            self.nodes.append(MapNode(i, name, crd, is_station))
        
        for i, node_info in enumerate(map_data):
            neighbors = [int(x) for x in node_info[-1].split(",")]
            for j in neighbors:
                self.nodes[i].add_neighbor(self.nodes[j])
        
        self.nodes.sort(key=lambda x: x.name)
        for node in self.nodes:
            if node.name in vertices_names:
                self.vertex_inds.append(node)
            if node.is_station:
                self.station_inds.append(node)
        for i, node in enumerate(self.nodes):
            node.ind = i
        
        print("Map loaded from {} with {} nodes".format(csv_fl, len(self.nodes)))

    def discreatize(self, granularity=150):
        # Discreatize the edge weights
        edges_weights = {}
        for s_i in self.nodes:
            for s_j in s_i.neighbors:
                start_ind, end_ind = min([s_i.ind, s_j.ind]), max([s_i.ind, s_j.ind])
                edge_name = 'e_{}_{}'.format(start_ind, end_ind)
                dist = np.linalg.norm(s_i.crd - s_j.crd)
                edges_weights[edge_name] = [start_ind, end_ind, max(1, int(np.round(dist / granularity)))]
        
        for edge in sorted(list(edges_weights.keys())):
            start_ind, end_ind, dist = edges_weights[edge]
            if dist == 1:
                continue
            # add vertices, make an equivalent unweighted graph
            start_node = self.nodes[start_ind]
            end_node = self.nodes[end_ind]

            new_nodes = [start_node]
            for i in range(dist - 1):
                ind = len(self.nodes) + i
                crd = start_node.crd * (1 - (i + 1) / dist) + end_node.crd * ((i + 1) / dist)
                new_node = MapNode(ind, edge + '_n{}'.format(i), crd, is_station=False)
                new_node.add_neighbor(new_nodes[-1])
                new_nodes.append(new_node)
            
            start_node.remove_neighbor(end_node)
            end_node.remove_neighbor(start_node)
            end_node.add_neighbor(new_nodes[-1])
            new_nodes[-1].add_neighbor(end_node)

            for i, node in enumerate(new_nodes[:-1]):
                node.add_neighbor(new_nodes[i + 1])
            
            self.nodes += new_nodes[1:]

    def get_paths(self, power=False):

        def all_paths(s, e):
            visited = [False] * len(self.nodes)
            path = []
            path_record = []
            all_paths_utils(s, e, visited, path, path_record)
            return path_record

        print("Stations:")
        station_inds, vertex_inds = [], []
        for s in self.station_inds:
            print(" * ", s)
            station_inds.append(s.ind)

        for s in self.vertex_inds:
            vertex_inds.append(s.ind)

        self.routes_dict = {}
        for s in self.vertex_inds:
            for e in self.vertex_inds:
                if s.ind == e.ind:
                    continue
                paths = all_paths(s, e)
                self.routes_dict.update(gen_routes(paths, station_inds, vertex_inds, power))
        
        self.route_names = sorted(list(self.routes_dict.keys()))
        self.routes = []
        print("Total Lines {}".format(len(self.route_names)))
        self.routes_choice = {i: [] for i in vertex_inds}
        for i, route_name in enumerate(self.route_names):
            s_ind = int(route_name.split("_")[0])
            self.routes_choice[s_ind].append(i)
            self.routes.append(self.routes_dict[route_name])

    def plot_simple(self):
        plt.figure(figsize=(12, 6))
        for node in self.nodes:
            for neighbor in node.neighbors:
                plt.plot([node.crd[0], neighbor.crd[0]], [node.crd[1], neighbor.crd[1]], c='b')
            plt.scatter([node.crd[0]], [node.crd[1]], color='r' if node.is_station else 'g')
            if node.is_station:
                plt.text(node.crd[0], node.crd[1], node.name, ma='right')
        plt.gca().set_aspect('equal')
        plt.savefig('sample.png')

        