import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, VGAE
from torch_geometric.utils import to_undirected, subgraph, add_self_loops
from torch.nn import Sequential, Linear, ReLU


class GIN_NodeWeightEncoder(nn.Module):
    """节点权重编码器"""

    def __init__(self, in_channels, dim, add_mask=False):
        super().__init__()
        self.add_mask = add_mask

        # 特征提取层
        nn1 = Sequential(
            Linear(in_channels, dim),
            ReLU(),
            Linear(dim, dim)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)

        # 输出层
        out_dim = 3 if add_mask else 2
        nn2 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, out_dim)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()

        # 特征提取
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        # 节点权重预测
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        return x


class ViewGenerator(VGAE):
    """视图生成器"""

    def __init__(self, input_dim, dim, encoder, add_mask=False):
        super().__init__(encoder=encoder(input_dim + 3, dim, add_mask))
        self.input_dim = input_dim
        self.add_mask = add_mask

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim + 3, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim + 3)
        )

    def forward(self, data_in, requires_grad=True, current_epoch=None):
        data = copy.deepcopy(data_in)
        x, edge_index = data.x, data.edge_index

        # 确保中心性特征存在
        if not hasattr(data, 'centrality'):
            print("警告：缺少中心性特征，使用零张量替代")
            centrality = torch.zeros(x.size(0), 3, device=x.device)
        else:
            centrality = data.centrality

        # 打印特征形状进行验证
        print(f"输入特征形状: {x.shape}, 中心性特征形状: {centrality.shape}")

        # 特征融合
        combined = torch.cat([x, centrality], dim=1)
        fused = self.feature_fusion(combined)
        data.x = fused

        # 生成节点权重
        p = self.encoder(data)
        tau = max(0.1, 1.0 - (current_epoch or 0) / 100)  # 动态温度参数
        sample = F.gumbel_softmax(p, tau=tau, hard=True)

        # 节点选择和特征掩码
        keep_sample = sample[:, 0] + (sample[:, 2] if self.add_mask else 0)
        keep_idx = torch.where(keep_sample > 0.5)[0]

        # 子图提取
        if keep_idx.numel() > 0:  # 确保有节点被保留
            edge_index, _ = subgraph(keep_idx, edge_index, num_nodes=data.num_nodes)
        else:
            # 如果没有节点被保留，创建一个空图
            edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        data.edge_index = edge_index

        # 动态调整的正则损失
        epoch_factor = min(1.0, (current_epoch or 0) / 100)
        centrality_loss = F.mse_loss(
            keep_sample.detach(),
            centrality.mean(dim=1)
        ) * 0.05 * epoch_factor

        return keep_sample, data, centrality_loss