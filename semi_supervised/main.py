import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
import logging
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from train_eval import benchmark_exp, joint_cl_exp
from res_gcn import ResGCN
from view_generator import ViewGenerator, GIN_NodeWeightEncoder
from tqdm import tqdm


def precompute_centrality(dataset):
    """预计算中心性特征"""
    for data in tqdm(dataset, desc="Precomputing centrality"):
        try:
            G = to_networkx(data, to_undirected=True)

            # 度中心性
            deg_cent = nx.degree_centrality(G)

            # 接近中心性
            try:
                close_cent = nx.closeness_centrality(G)
            except:
                close_cent = {n: 0.0 for n in G.nodes}

            # 近似中介中心性
            between_cent = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))

            # 转换为tensor并归一化
            centrality = torch.tensor([
                [deg_cent[i], close_cent[i], between_cent[i]]
                for i in range(data.num_nodes)
            ], dtype=torch.float)

            # 归一化
            mean = centrality.mean(dim=0, keepdim=True)
            std = centrality.std(dim=0, keepdim=True) + 1e-8
            data.centrality = (centrality - mean) / std

        except Exception as e:
            print(f"中心性计算失败: {str(e)}")
            data.centrality = torch.zeros(data.num_nodes, 3)

    return dataset


def run_benchmark_exp(args, device, logger):
    """运行基准测试实验"""
    dataset = TUDataset(root=args.data_root, name=args.dataset)

    # 获取输入维度和类别数
    input_dim = dataset.num_features
    num_classes = dataset.num_classes

    # 创建模型实例
    model = ResGCN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden=args.hidden,
        num_conv_layers=args.n_conv_layers,
        num_fc_layers=2,
        global_pool="sum",
        dropout=0.5
    )

    # 创建模型函数（直接返回模型实例）
    model_func = lambda: model

    train_acc, acc, std, duration = benchmark_exp(
        device,
        logger,
        dataset,
        model_func,
        folds=args.n_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=args.weight_decay,
        epoch_select=args.epoch_select,
        with_eval_mode=args.with_eval_mode,
        semi_split=args.semi_split
    )

    if logger:
        logger.info(
            f'Final Result - Train Acc: {train_acc:.4f}, Test Acc: {acc:.3f} ± {std:.3f}, Duration: {duration:.3f}')
    else:
        print(f'Final Result - Train Acc: {train_acc:.4f}, Test Acc: {acc:.3f} ± {std:.3f}, Duration: {duration:.3f}')

    return train_acc, acc, std, duration


def run_joint_cl_exp(args, device, logger):
    """运行联合对比学习实验"""
    dataset = TUDataset(root=args.data_root, name=args.dataset)
    # 预计算中心性
    dataset = precompute_centrality(dataset)

    # 获取输入维度和类别数
    num_classes = dataset.num_classes

    # 修改为接受输入维度的模型函数
    def model_func(input_dim):
        return ResGCN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden=args.hidden,
            num_conv_layers=args.n_conv_layers,
            num_fc_layers=2,
            global_pool="sum",
            dropout=0.5
        )

    train_acc, acc, std, duration = joint_cl_exp(
        device,
        logger,
        dataset,
        model_func,  # 直接传递模型函数
        folds=args.n_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=args.weight_decay,
        epoch_select=args.epoch_select,
        with_eval_mode=args.with_eval_mode,
        semi_split=args.semi_split,
        add_mask=args.add_mask
    )

    if logger:
        logger.info(
            f'Final Result - Train Acc: {train_acc:.4f}, Test Acc: {acc:.3f} ± {std:.3f}, Duration: {duration:.3f}')
    else:
        print(f'Final Result - Train Acc: {train_acc:.4f}, Test Acc: {acc:.3f} ± {std:.3f}, Duration: {duration:.3f}')

    return train_acc, acc, std, duration


def setup_logger(save_dir):
    """设置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    log_file = os.path.join(save_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp', type=str, default="benchmark", choices=["benchmark", "joint_cl_exp"])
    parser.add_argument('--data_root', type=str, default="../data")
    parser.add_argument('--dataset', type=str, default="MUTAG")
    parser.add_argument('--gpu', type=int, default=0)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_fold', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epoch_select', type=str, default='test_max')

    # 模型参数
    parser.add_argument('--hidden', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--n_conv_layers', type=int, default=3, help='卷积层数量')

    # 实验选项
    parser.add_argument('--add_mask', action='store_true', help='是否添加特征掩码')
    parser.add_argument('--save', type=str, default="results", help='结果保存路径')
    parser.add_argument('--with_eval_mode', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='评估时是否使用eval模式')
    parser.add_argument('--semi_split', type=int, default=10,
                        help='半监督训练数据百分比')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备设置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save, exist_ok=True)

    # 设置日志
    logger = setup_logger(args.save)
    logger.info("实验参数:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # 运行实验
    if args.exp == 'benchmark':
        results = run_benchmark_exp(args, device, logger)
    elif args.exp == 'joint_cl_exp':
        results = run_joint_cl_exp(args, device, logger)

    # 保存结果
    results_df = pd.DataFrame([{
        'dataset': args.dataset,
        'exp_type': args.exp,
        'train_acc': results[0],
        'test_acc': results[1],
        'std': results[2],
        'duration': results[3],
        'seed': args.seed,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'hidden': args.hidden,
        'n_conv_layers': args.n_conv_layers
    }])
    results_path = os.path.join(args.save, f'results_{args.dataset}_{args.exp}.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"实验结果保存至 {results_path}")