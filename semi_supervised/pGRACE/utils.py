import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.sparse.linalg import eigsh
import os.path as osp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
# from libKMCUDA import kmeans_cuda
from tqdm import tqdm
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx
from torch_geometric.datasets import KarateClub
from torch_scatter import scatter
import torch_sparse

import networkx as nx
import matplotlib.pyplot as plt


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


# def precompute_centrality(dataset):
#     """预计算中心性特征并添加到数据集"""
#     for data in tqdm(dataset, desc="Precomputing centrality"):
#         G = to_networkx(data, to_undirected=True)
#
#         # 度中心性
#         deg_cent = nx.degree_centrality(G)
#
#         # 接近中心性（仅计算最大连通分量）
#         if nx.is_connected(G):
#             close_cent = nx.closeness_centrality(G)
#         else:
#             close_cent = {n: 0 for n in G.nodes}
#
#         # 近似中介中心性
#         between_cent = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
#
#         # 转换为tensor并归一化
#         centrality = torch.tensor([
#             [deg_cent[i], close_cent[i], between_cent[i]]
#             for i in range(data.num_nodes)
#         ], dtype=torch.float)
#         centrality = (centrality - centrality.mean(dim=0)) / (centrality.std(dim=0) + 1e-8)
#
#         data.centrality = centrality
#
#     return dataset