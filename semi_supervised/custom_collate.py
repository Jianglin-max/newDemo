import torch
from torch_geometric.data import Data, Batch


def custom_collate(batch):
    """自定义数据批处理函数，处理中心性特征"""
    batch = Batch.from_data_list(batch)

    # 确保中心性特征被正确传递
    if hasattr(batch, 'centrality'):
        centrality_list = [data.centrality for data in batch.to_data_list()]
        batch.centrality = torch.cat(centrality_list, dim=0)

    return batch