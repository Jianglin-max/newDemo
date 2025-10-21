import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv


class ResGCN(nn.Module):
    """动态适应输入维度的GCN模型"""

    def __init__(self, input_dim, num_classes, hidden=128, num_conv_layers=3,
                 num_fc_layers=2, global_pool="sum", dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden = hidden

        # 输入特征处理层
        self.bn_feat = BatchNorm1d(input_dim)
        self.conv_feat = GCNConv(input_dim, hidden)

        # 卷积层
        self.convs = nn.ModuleList()
        self.bns_conv = nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns_conv.append(BatchNorm1d(hidden))

        # 池化设置
        assert global_pool in ["sum", "mean"], "不支持的池化方式"
        self.global_pool = global_add_pool if global_pool == "sum" else global_mean_pool

        # 门控机制
        self.gating = nn.Sequential(
            Linear(hidden, hidden),
            nn.ReLU(),
            Linear(hidden, 1),
            nn.Sigmoid()
        )

        # 全连接层
        self.fcs = nn.ModuleList()
        self.bns_fc = nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.fcs.append(Linear(hidden, hidden))
            self.bns_fc.append(BatchNorm1d(hidden))

        self.classifier = Linear(hidden, num_classes)
        self.dropout = dropout

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)
            elif isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 确保输入维度匹配
        if x.size(1) != self.input_dim:
            new_x = torch.zeros(x.size(0), self.input_dim, device=x.device)
            min_dim = min(x.size(1), self.input_dim)
            new_x[:, :min_dim] = x[:, :min_dim]
            x = new_x

        # 特征标准化 (跳过单样本的BatchNorm)
        if x.size(0) > 1:
            x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        # 卷积层
        for conv, bn in zip(self.convs, self.bns_conv):
            x = F.relu(conv(bn(x), edge_index))

        # 门控池化
        gate = self.gating(x)
        x = self.global_pool(x * gate, batch)

        # 全连接层
        for fc, bn in zip(self.fcs, self.bns_fc):
            x = F.relu(fc(bn(x)))

        # 分类器
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

    def forward_cl(self, data):
        """用于对比学习的特征提取"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 确保输入维度匹配
        if x.size(1) != self.input_dim:
            new_x = torch.zeros(x.size(0), self.input_dim, device=x.device)
            min_dim = min(x.size(1), self.input_dim)
            new_x[:, :min_dim] = x[:, :min_dim]
            x = new_x

        # 特征标准化 (跳过单样本的BatchNorm)
        if x.size(0) > 1:
            x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        # 卷积层
        for conv, bn in zip(self.convs, self.bns_conv):
            x = F.relu(conv(bn(x), edge_index))

        # 门控池化
        gate = self.gating(x)
        x = self.global_pool(x * gate, batch)

        # 全连接层
        for fc, bn in zip(self.fcs, self.bns_fc):
            x = F.relu(fc(bn(x)))

        return x