import logging
import sys
import time
import copy

import networkx as nx
from nni_assets.hello_hpo.model import epochs
from sklearn.model_selection import StratifiedKFold
import random
import argparse
import os

import torch
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from gca import get_gca_param, get_gca_model, train_gca
from gca import GCA_Classifier, train_gca_cls, eval_gca_acc

sys.path.append(os.path.abspath(os.path.join('..')))
from view_generator import ViewGenerator, GIN_NodeWeightEncoder
from augs import Augmentor
from utils import print_weights

from IPython import embed


def loss_cl(x1, x2, temperature=0.5):
    """改进的对比损失函数"""
    x1 = F.normalize(x1, p=2, dim=1)
    x2 = F.normalize(x2, p=2, dim=1)

    logits = torch.mm(x1, x2.t()) / temperature
    labels = torch.arange(x1.size(0), device=x1.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_j) / 2


def get_snapshot(view_gen1, view_gen2, model):
    snapshot = {
        'view_gen1': copy.deepcopy(view_gen1.state_dict()),
        'view_gen2': copy.deepcopy(view_gen2.state_dict()),
        'model': copy.deepcopy(model.state_dict())
    }
    return snapshot

def load_snapshot(snapshot, view_gen1, view_gen2, model):
    view_gen1.load_state_dict(snapshot['view_gen1'])
    view_gen2.load_state_dict(snapshot['view_gen2'])
    model.load_state_dict(snapshot['model'])

def benchmark_exp(device, logger, dataset, model_func,
                 folds, epochs, batch_size,
                 lr, lr_decay_factor, lr_decay_step_size, weight_decay,
                 epoch_select, with_eval_mode=True, semi_split=None):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):

        # 动态获取特征维度
        sample_data = dataset[0]
        input_dim = sample_data.x.size(1)

        # 修复：直接调用 model_func 而不传递参数
        model = model_func()  # 移除 dataset 参数
        model.to(device)

        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # train_dataset = dataset[train_idx]
        semi_dataset = dataset[semi_idx]
        # val_dataset = dataset[val_idx]
        # test_dataset = dataset[test_idx]
        test_dataset = dataset[test_idx.long()]

        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        # # Change to GIN model
        # model = GIN_Classifier(dataset, 128)
        # model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Pre-training Classifier...")
        # best_model = None
        for epoch in range(1, epochs + 1):
            train_loss = train_cls(model, optimizer, semi_loader, device)
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss,
                                                                train_acc, test_acc))

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
                    # logger.info("lr:" + str(param_group['lr']))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


def aug_only_exp(args, device, logger, dataset, model_func,
                 folds, epochs, batch_size,
                 lr, lr_decay_factor, lr_decay_step_size, weight_decay,
                 epoch_select, with_eval_mode=True, semi_split=None):
    # === GPU兼容的中心性计算函数 ===
    def precompute_centrality(data):
        # 临时复制到CPU计算（NetworkX只支持CPU）
        cpu_data = data.clone().to('cpu')
        G = nx.Graph()

        # 安全处理边索引
        if cpu_data.edge_index.nelement() == 0:
            # 没有边的图（如孤立节点）
            edges = []
        else:
            edges = cpu_data.edge_index.numpy().T

        try:
            G.add_edges_from(edges)

            # 计算中心性指标并处理可能的错误
            try:
                deg = list(nx.degree_centrality(G).values())
            except:
                deg = [1.0] * cpu_data.num_nodes

            try:
                closeness = list(nx.closeness_centrality(G).values())
            except:
                closeness = [1.0] * cpu_data.num_nodes

            try:
                betweenness = list(nx.betweenness_centrality(G).values())
            except:
                betweenness = [1.0] * cpu_data.num_nodes

        except nx.NetworkXError:
            # 网络不连通时的回退方案
            deg = closeness = betweenness = [1.0] * cpu_data.num_nodes

        # 创建一个3维的中心性特征向量
        centrality = torch.tensor([
            deg,
            closeness,
            betweenness
        ], dtype=torch.float).t()  # 转置为 [节点数, 3]

        # 移动到原始数据设备（可能是GPU）
        centrality = centrality.to(data.x.device)
        data.centrality = centrality
        return data

    # === 关键修改：预处理中心性特征 ===
    # 移动整个数据集到目标设备后再计算中心性
    dataset = [data.to(device) for data in dataset]
    dataset = [precompute_centrality(data) for data in dataset]
    # ===========================================

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    fold_data = cl_k_fold(dataset, folds, epoch_select, semi_split)
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*fold_data)):

        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # 创建数据集子集
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]
        val_dataset = [dataset[i] for i in val_idx]
        semi_dataset = [dataset[i] for i in semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Semi size: %d" % len(semi_dataset))
        logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        # === 关键修改：初始化视图生成器（添加add_mask参数）===
        # 使用args中的add_mask参数
        view_gen1 = ViewGenerator(dataset[0].num_features, 128, GIN_NodeWeightEncoder, add_mask=args.add_mask)
        view_gen2 = ViewGenerator(dataset[0].num_features, 128, GIN_NodeWeightEncoder, add_mask=args.add_mask)
        view_gen1 = view_gen1.to(device)
        view_gen2 = view_gen2.to(device)

        # 优化器设置
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        view_optimizer = Adam([
            {'params': view_gen1.parameters()},
            {'params': view_gen2.parameters()}
        ], lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training AutoGCL with centrality features...")
        best_test_acc = 0

        for epoch in range(1, epochs + 1):
            # 确保训练函数正确处理设备
            train_loss, sim_loss, cls_loss = train_cls_with_node_weight_view_gen(
                view_gen1, view_gen2, view_optimizer, model, optimizer, semi_loader, device
            )

            # 评估模型
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            val_acc = eval_acc(model, val_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)

            # 记录最佳测试精度
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                        'Sim Loss: {:.4f}, Cls Loss: {:.4f}, '
                        'Train Acc: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'.format(
                epoch, train_loss, sim_loss, cls_loss, train_acc, val_acc, test_acc))

            # 学习率衰减
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
                for param_group in view_optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        # 保存结果
        train_accs.append(train_acc)
        test_accs.append(best_test_acc)
        logger.info('Fold {} - Best Test Acc: {:.4f}'.format(fold, best_test_acc))

    # 计算平均结果
    duration = torch.tensor(durations)
    train_acc = torch.tensor(train_accs)
    test_acc = torch.tensor(test_accs)

    train_acc_mean = train_acc.mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info('Final Result - Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.format(
        train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

def train_cls_with_node_weight_view_gen(view_gen1, view_gen2, view_optimizer, model, optimizer, loader, device):
    model.train()
    view_gen1.train()
    view_gen2.train()

    total_loss = 0
    total_sim_loss = 0
    total_cls_loss = 0

    for data in loader:
        # === 关键修复：确保数据在正确设备上 ===
        data = data.to(device)

        # 清零梯度
        view_optimizer.zero_grad()
        optimizer.zero_grad()

        # 生成两个视图
        _, view1, centrality_loss1 = view_gen1(data, requires_grad=True)
        _, view2, centrality_loss2 = view_gen2(data, requires_grad=True)

        # 计算对比损失
        z1 = model.forward_cl(view1)
        z2 = model.forward_cl(view2)

        # 这里使用简单的MSE作为示例，实际中可用InfoNCE等损失
        sim_loss = F.mse_loss(z1, z2)

        # 分类损失
        output = model(data)
        cls_loss = F.nll_loss(output, data.y)

        # 总损失（包含中心性正则项）
        total_loss_batch = sim_loss + cls_loss + centrality_loss1 + centrality_loss2

        # 反向传播
        total_loss_batch.backward()
        view_optimizer.step()
        optimizer.step()

        # 记录损失
        total_loss += total_loss_batch.item()
        total_sim_loss += sim_loss.item()
        total_cls_loss += cls_loss.item()

    n = len(loader)
    return total_loss / n, total_sim_loss / n, total_cls_loss / n
def get_snapshot(view_gen1, view_gen2, model):
    """保存模型状态"""
    return {
        'view_gen1': copy.deepcopy(view_gen1.state_dict()),
        'view_gen2': copy.deepcopy(view_gen2.state_dict()),
        'model': copy.deepcopy(model.state_dict())
    }

def load_snapshot(snapshot, view_gen1, view_gen2, model):
    """加载模型状态"""
    view_gen1.load_state_dict(snapshot['view_gen1'])
    view_gen2.load_state_dict(snapshot['view_gen2'])
    model.load_state_dict(snapshot['model'])

def cl_gca_exp(device, logger, dataset, model_func, folds, epochs, batch_size,
            lr, lr_decay_factor, lr_decay_step_size, weight_decay, epoch_select,
            with_eval_mode=True, semi_split=None):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    gca_args, param = get_gca_param()

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):

        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size * 4, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model, encoder = get_gca_model(dataset, param, gca_args)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param['learning_rate'],
            weight_decay=param['weight_decay']
        )

        # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training View Generator and Classifier...")
        best_test_acc = 0
        after_train_view_snapshot = None

        for epoch in range(1, 30 + 1):
            cl_loss = train_gca(train_loader, param, model, optimizer, device, gca_args)
            logger.info('Epoch: {:03d}, CL Loss: {:.4f}'.format(epoch, cl_loss))

        gca_cls = GCA_Classifier(dataset, param["num_hidden"])
        gca_cls.to(device)
        gca_cls_optimizer = Adam([
                                    {"params": gca_cls.parameters()},
                                    {"params": model.parameters()}
                                ], lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs + 1):
            cls_loss = train_gca_cls(semi_loader, model, gca_cls, gca_cls_optimizer, device)

            train_acc = eval_gca_acc(model, gca_cls, semi_loader, device, with_eval_mode)
            test_acc = eval_gca_acc(model, gca_cls, test_loader, device, with_eval_mode)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            logger.info('Epoch: {:03d}, Cls Loss: {:.4f}, '
                        'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, cls_loss,
                                                                    train_acc, test_acc))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))

    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

# GraphCL Reproduced
def graph_cl_exp(
            device, logger, dataset, model_func, folds, epochs, batch_size,
            lr, lr_decay_factor, lr_decay_step_size, weight_decay, epoch_select,
            with_eval_mode=True, semi_split=None, aug_ratio=0.2):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):

        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        augmentor = Augmentor(aug_ratio)

        logger.info("*" * 50)
        logger.info("Training Contrastive learning...")
        for epoch in range(1, epochs + 1):
            train_loss = train_graph_cl(augmentor, model, optimizer, train_loader, device)
            logger.info('Epoch: {:03d}, CL Loss: {:.4f}'.format(epoch, train_loss))

        logger.info("*" * 50)
        logger.info("Pretraining Classifier...")
        best_model = None
        best_test_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = train_cls(model, optimizer, semi_loader, device)
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss,
                                                                train_acc, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model)
        best_model = None

        logger.info("*" * 50)
        logger.info("Fold: {}, Best Test Acc: {:.4f}".format(fold, best_test_acc))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    # logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

# GraphCL reproduced with only the augmentations
def graph_cl_aug_only_exp(
            device,
            logger,
            dataset,
            model_func,
            folds,
            epochs,
            batch_size,
            lr,
            lr_decay_factor,
            lr_decay_step_size,
            weight_decay,
            epoch_select,
            with_eval_mode=True,
            semi_split=None,
            aug_ratio=0.2):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    # print("developing CL...")
    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):

        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        t_start = time.perf_counter()

        augmentor = Augmentor(aug_ratio)

        logger.info("*" * 50)
        logger.info("Training Classifier with GraphCL Augs...")
        best_model = None
        best_test_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = train_graph_cl_aug_semi(augmentor, model, optimizer, semi_loader, device)
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss,
                                                                train_acc, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model)
        best_model = None

        logger.info("*" * 50)
        logger.info("Fold: {}, Best Test Acc: {:.4f}".format(fold, best_test_acc))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    # logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


def naive_cl_exp(device, logger, dataset, model_func, folds, epochs, batch_size,
            lr, lr_decay_factor, lr_decay_step_size, weight_decay, epoch_select,
            with_eval_mode=True, semi_split=None):

    assert epoch_select in ['val_max', 'test_max'], epoch_select
    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):

        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size * 4, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        view_gen1 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
        view_gen2 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
        view_gen1 = view_gen1.to(device)
        view_gen2 = view_gen2.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        view_optimizer = Adam([ {'params': view_gen1.parameters()},
                                {'params': view_gen2.parameters()} ], lr=lr
                                , weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training View Generator and Classifier...")
        best_test_acc = 0
        after_train_view_snapshot = None

        for epoch in range(1, epochs + 1):
            cl_loss = train_cl_cls_with_fix_node_weight_view_gen(view_gen1, view_gen2, model,
                                                    optimizer, train_loader, device)
            logger.info('Epoch: {:03d}, CL Loss: {:.4f}'.format(epoch, cl_loss))

        for epoch in range(1, epochs + 1):
            train_view_loss, sim_loss, cls_loss = train_node_weight_view_gen_and_cls(
                                                    view_gen1, view_gen2,
                                                    view_optimizer,
                                                    model, optimizer,
                                                    semi_loader, device)

            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            logger.info('Epoch: {:03d}, Train View Loss: {:.4f}, Sim Loss: {:.4f}, '
                    'Cls Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_view_loss, sim_loss,
                                                                cls_loss, train_acc, test_acc))

            if epoch % lr_decay_step_size == 0:
                for param_group in view_optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))

    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


# 修改函数签名
def joint_cl_exp(
    device,
    logger,
    dataset,
    model_func,  # 现在是一个接受输入维度的函数
    folds,
    epochs,
    batch_size,
    lr,
    lr_decay_factor,
    lr_decay_step_size,
    weight_decay,
    epoch_select,
    with_eval_mode=True,
    semi_split=None,
    add_mask=False
):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []

    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):

        # 动态获取特征维度
        sample_data = dataset[0]
        input_dim = sample_data.x.size(1)

        # 修复：直接调用 model_func 而不传递参数
        model = model_func()  # 移除 dataset 参数
        model.to(device)

        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # 确保索引是torch.long类型
        train_idx = torch.as_tensor(train_idx, dtype=torch.long)
        test_idx = torch.as_tensor(test_idx, dtype=torch.long)
        val_idx = torch.as_tensor(val_idx, dtype=torch.long)
        semi_idx = torch.as_tensor(semi_idx, dtype=torch.long)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        semi_dataset = dataset[semi_idx]

        # 创建 DataLoader 时不使用 collate_fn
        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size * 4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        # 动态获取特征维度
        if len(semi_dataset) > 0:
            sample_data = semi_dataset[0]
        else:
            sample_data = dataset[0]
        num_features = sample_data.num_features

        # 计算扩展后的特征维度
        expanded_dim = num_features + 3  # 原始特征 + 3维中心性
        logger.info(f"Fold {fold}: Using feature dimension: {num_features}, Expanded to: {expanded_dim}")

        # === 关键修复：传入扩展后的维度 ===
        model = model_func(expanded_dim)  # 这里传入计算出的维度
        model.to(device)

        # 使用动态特征维度创建ViewGenerator
        view_gen1 = ViewGenerator(num_features, 128, GIN_NodeWeightEncoder, add_mask)
        view_gen2 = ViewGenerator(num_features, 128, GIN_NodeWeightEncoder, add_mask)
        view_gen1 = view_gen1.to(device)
        view_gen2 = view_gen2.to(device)

        # ... [后续代码保持不变] ...

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        view_optimizer = Adam([
            {'params': view_gen1.parameters()},
            {'params': view_gen2.parameters()}
        ], lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training View Generator and Classifier...")

        for epoch in range(1, epochs + 1):
            # === 修改调用方式 ===
            train_view_loss, sim_loss, cls_loss, cl_loss = \
                train_node_weight_view_gen_and_cls(
                    view_gen1, view_gen2, view_optimizer,
                    model, optimizer, semi_loader, device,
                    epoch  # 传递当前epoch数
                )

            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            logger.info('Epoch: {:03d}, Train View Loss: {:.4f}, Sim Loss: {:.4f}, '
                        'Cls Loss: {:.4f}, CL Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(
                epoch, train_view_loss, sim_loss,
                cls_loss, cl_loss, train_acc, test_acc))

            if epoch % lr_decay_step_size == 0:
                for param_group in view_optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.format(
        train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


def train_node_weight_view_gen_and_cls(view_gen1, view_gen2, view_optimizer,
                                       model, optimizer, loader, device, epoch=None):
    view_gen1.train()
    view_gen2.train()
    model.train()

    view_loss_total = 0
    sim_loss_total = 0
    cls_loss_total = 0
    total_graphs = 0

    for data in loader:
        data = data.to(device)

        # 步骤1: 更新视图生成器
        view_optimizer.zero_grad()
        _, view1, closs1 = view_gen1(data, True, epoch)
        _, view2, closs2 = view_gen2(data, True, epoch)

        # 对比损失
        z1 = model.forward_cl(view1)
        z2 = model.forward_cl(view2)
        sim_loss = loss_cl(z1, z2)

        # 视图生成器总损失
        view_loss = sim_loss + closs1 + closs2
        view_loss.backward()
        view_optimizer.step()

        # 步骤2: 更新分类器
        optimizer.zero_grad()

        # 使用原始数据和生成视图
        output_original = model(data)
        output_view1 = model(view1.detach())
        output_view2 = model(view2.detach())

        # 组合分类损失
        cls_loss = (
                           F.nll_loss(output_original, data.y) +
                           F.nll_loss(output_view1, data.y) +
                           F.nll_loss(output_view2, data.y)
                   ) / 3

        cls_loss.backward()
        optimizer.step()

        # 统计损失
        num_graphs = data.num_graphs
        view_loss_total += view_loss.item() * num_graphs
        sim_loss_total += sim_loss.item() * num_graphs
        cls_loss_total += cls_loss.item() * num_graphs
        total_graphs += num_graphs

    # 计算平均损失
    view_loss_avg = view_loss_total / total_graphs if total_graphs > 0 else 0
    sim_loss_avg = sim_loss_total / total_graphs if total_graphs > 0 else 0
    cls_loss_avg = cls_loss_total / total_graphs if total_graphs > 0 else 0

    return view_loss_avg, sim_loss_avg, cls_loss_avg


def train_cl_cls_with_fix_node_weight_view_gen(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    total_graphs = 0

    for data in loader:
        optimizer.zero_grad()

        data = data.to(device)
        _, view1 = view_gen1(data, False)
        _, view2 = view_gen2(data, False)

        input_list = [data, view1, view2]
        # input_list = [view1, view2]
        input1, input2 = random.choices(input_list, k=2)

        # embed()
        # exit()

        output1 = model.forward_cl(input1)
        output2 = model.forward_cl(input2)

        cl_loss = loss_cl(output1, output2)

        loss = cl_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        optimizer.step()

    loss_all /= total_graphs

    return loss_all

def train_graph_cl(augmentor, model, optimizer, loader, device):
    model.train()

    loss_all = 0
    total_graphs = 0

    for data in loader:
        optimizer.zero_grad()

        data = data.to(device)
        view1 = augmentor(data)
        view2 = augmentor(data)

        output1 = model.forward_cl(view1)
        output2 = model.forward_cl(view2)

        cl_loss = loss_cl(output1, output2)
        loss = cl_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        optimizer.step()

    loss_all /= total_graphs
    return loss_all

def train_graph_cl_aug_semi(augmentor, model, optimizer, loader, device):
    model.train()

    loss_all = 0
    total_graphs = 0

    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        view = augmentor(data)
        output = model(view)
        loss = F.nll_loss(output, data.y)
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        optimizer.step()

    loss_all /= total_graphs
    return loss_all

def train_cl(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    total_graphs = 0

    for data in loader:
        optimizer.zero_grad()

        data = data.to(device)
        # z1, rec1, view1 = view_gen1(data)
        # z2, rec2, view2 = view_gen2(data)

        # output1 = model.forward_cl(view1)
        # output2 = model.forward_cl(view2)
        # raw cl
        output1 = model.forward_cl(data)
        output2 = output1

        cl_loss = loss_cl(output1, output2)
        loss = cl_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        optimizer.step()

    loss_all /= total_graphs
    return loss_all

def train_cls_with_fix_view_gen(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    cls_loss_all = 0
    total_graphs = 0

    for data in loader:
        optimizer.zero_grad()

        data = data.to(device)
        _, _, view1 = view_gen1(data)
        _, _, view2 = view_gen2(data)

        output1 = model(view1)
        output2 = model(view2)

        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)

        cls_loss = (loss1 + loss2) / 2
        loss = cls_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        optimizer.step()

    loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all


def train_cls_with_node_weight_view_gen(view_gen1, view_gen2, view_optimizer, model, optimizer, loader, device):
    model.train()
    view_gen1.train()
    view_gen2.train()

    total_loss = 0
    total_sim_loss = 0
    total_cls_loss = 0

    for data in loader:
        data = data.to(device)

        # === 关键修复：启用梯度并获取中心性损失 ===
        _, view1, centrality_loss1 = view_gen1(data, requires_grad=True)
        _, view2, centrality_loss2 = view_gen2(data, requires_grad=True)

        # 计算对比损失
        z1 = model.forward_cl(view1)
        z2 = model.forward_cl(view2)
        sim_loss = loss_cl(z1, z2)

        # 计算分类损失（原始数据）
        cls_out = model(data)
        cls_loss = F.nll_loss(cls_out, data.y)

        # === 关键修复：整合所有损失 ===
        total_batch_loss = sim_loss + cls_loss + centrality_loss1 + centrality_loss2

        # 反向传播
        optimizer.zero_grad()
        view_optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        view_optimizer.step()

        total_loss += total_batch_loss.item()
        total_sim_loss += sim_loss.item()
        total_cls_loss += cls_loss.item()

    return total_loss / len(loader), total_sim_loss / len(loader), total_cls_loss / len(loader)

def train_node_weight_view_gen_with_fix_cls(view_gen1, view_gen2, view_optimizer, model, loader, device):
    view_gen1.train()
    view_gen2.train()
    model.eval()
    # model.train()

    loss_all = 0
    sim_loss_all = 0
    cls_loss_all = 0
    total_graphs = 0

    for data in loader:
        view_optimizer.zero_grad()
        data = data.to(device)
        # output = model(data)

        sample1, view1 = view_gen1(data, True)
        sample2, view2 = view_gen2(data, True)

        sim_loss = F.l1_loss(sample1, sample2)
        sim_loss = torch.exp(1 - sim_loss)

        # output = model(data)
        output1 = model(view1)
        output2 = model(view2)

        # loss1 = F.nll_loss(output, data.y)
        # output = model(data)
        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)
        # loss1 = F.mse_loss(output1, output)
        # loss2 = F.mse_loss(output2, output)

        cls_loss = (loss1 + loss2) / 2
        # cls_loss = sim_loss

        # loss = cls_loss
        loss = sim_loss + cls_loss
        loss.backward()

        # embed()
        # exit()

        loss_all += loss.item() * data.num_graphs
        sim_loss_all += sim_loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()

    loss_all /= total_graphs
    sim_loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all, sim_loss_all, cls_loss_all
    # return loss_all, rec_loss_all, cls_loss_all

def train_cls_with_fix_node_weight_view_gen(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    cls_loss_all = 0
    total_graphs = 0

    for data in loader:
        optimizer.zero_grad()

        data = data.to(device)
        _, view1 = view_gen1(data, False)
        _, view2 = view_gen2(data, False)

        output1 = model(view1)
        output2 = model(view2)

        # output = model(data)
        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)

        # loss1 = F.mse_loss(output1, output)
        # loss2 = F.mse_loss(output2, output)

        cls_loss = (loss1 + loss2) / 2
        # cls_loss = loss1

        loss = cls_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        optimizer.step()

    loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all

def cl_k_fold(dataset, folds, epoch_select, semi_split):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []

    semi_indices = []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    dataset_size = len(dataset)
    semi_size = int(dataset_size * semi_split / 100)

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indice = torch.nonzero(train_mask, as_tuple=False).view(-1)
        train_indices.append(train_indice)

        # semi split
        train_size = train_indice.shape[0]
        select_idx = torch.randperm(train_size)[:semi_size]
        semi_indice = train_indice[select_idx]
        semi_indices.append(semi_indice)
    # embed()
    # exit()
    return train_indices, test_indices, val_indices, semi_indices

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        # embed()
        # exit()
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def train_cls(model, optimizer, loader, device):
    model.train()

    loss_all = 0
    total_graphs = 0

    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        optimizer.step()

    loss_all /= total_graphs
    return loss_all

@torch.no_grad()
def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


@torch.no_grad()
def eval_acc_with_view_gen(view_gen1, view_gen2, model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
        view_gen1.eval()
        view_gen2.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        _, _, view1 = view_gen1(data)
        _, _, view2 = view_gen2(data)

        with torch.no_grad():
            pred1 = model(view1).max(1)[1]
            pred2 = model(view2).max(1)[1]

        correct1 = pred1.eq(data.y.view(-1)).sum().item()
        correct2 = pred2.eq(data.y.view(-1)).sum().item()

        correct += (correct1 + correct2) / 2

    return correct / len(loader.dataset)

@torch.no_grad()
def eval_acc_with_node_weight_view_gen(view_gen1, view_gen2, model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
        view_gen1.eval()
        view_gen2.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        _, view1 = view_gen1(data, False)
        _, view2 = view_gen2(data, False)

        with torch.no_grad():
            pred1 = model(view1).max(1)[1]
            pred2 = model(view2).max(1)[1]

        correct1 = pred1.eq(data.y.view(-1)).sum().item()
        correct2 = pred2.eq(data.y.view(-1)).sum().item()

        correct += (correct1 + correct2) / 2

    return correct / len(loader.dataset)

def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
