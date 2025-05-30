"""
双重网络GNN书籍推荐系统主程序
整合数据预处理、图构建、模型训练和评估
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import argparse
import json
import warnings
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import preprocess_book_crossing_data
from graph_builder import build_graphs
from dual_network_gnn import DualNetworkGNN
from trainer import Trainer, DataSplitter
from evaluator import ModelEvaluator

warnings.filterwarnings('ignore')


class DualNetworkRecommendationSystem:
    """双重网络GNN推荐系统"""
    
    def __init__(self, config: Dict):
        """
        初始化推荐系统
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据
        self.processed_data = None
        self.ub_graph = None
        self.bb_graph = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # 模型
        self.model = None
        self.trainer = None
        self.evaluator = None
        
        # 结果
        self.training_history = None
        self.evaluation_results = None
        
        print(f"初始化推荐系统，设备: {self.device}")
    
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("\n" + "="*50)
        print("步骤1: 数据加载和预处理")
        print("="*50)
        
        self.processed_data = preprocess_book_crossing_data(
            data_path=self.config['data_path'],
            min_user_interactions=self.config['min_user_interactions'],
            min_book_interactions=self.config['min_book_interactions']
        )
        
        print(f"\n预处理完成:")
        print(f"  用户数: {self.processed_data['num_users']}")
        print(f"  书籍数: {self.processed_data['num_books']}")
        print(f"  交互数: {len(self.processed_data['ratings'])}")
        print(f"  用户特征维度: {self.processed_data['user_features'].shape[1]}")
        print(f"  书籍特征维度: {self.processed_data['book_features'].shape[1]}")
    
    def build_graphs(self):
        """构建双重图"""
        print("\n" + "="*50)
        print("步骤2: 图构建")
        print("="*50)
        
        self.ub_graph, self.bb_graph = build_graphs(
            processed_data=self.processed_data,
            ub_feedback_handling=self.config['ub_feedback_handling'],
            bb_similarity_method=self.config['bb_similarity_method'],
            bb_similarity_threshold=self.config['bb_similarity_threshold'],
            bb_max_connections=self.config['bb_max_connections']
        )
        
        print(f"\n图构建完成:")
        print(f"  用户-书籍图:")
        print(f"    节点类型: {list(self.ub_graph.node_types)}")
        print(f"    边类型: {list(self.ub_graph.edge_types)}")
        
        print(f"  书籍-书籍图:")
        print(f"    节点类型: {list(self.bb_graph.node_types)}")
        print(f"    边类型: {list(self.bb_graph.edge_types)}")
        
        # 打印边数量
        for edge_type in self.ub_graph.edge_types:
            edge_count = self.ub_graph[edge_type].edge_index.shape[1]
            print(f"    {edge_type}: {edge_count} 条边")
        
        for edge_type in self.bb_graph.edge_types:
            edge_count = self.bb_graph[edge_type].edge_index.shape[1]
            print(f"    {edge_type}: {edge_count} 条边")
    
    def split_data(self):
        """划分数据集"""
        print("\n" + "="*50)
        print("步骤3: 数据集划分")
        print("="*50)
        
        splitter = DataSplitter(
            split_method=self.config['split_method'],
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            test_ratio=self.config['test_ratio'],
            random_state=self.config['random_state']
        )
        
        self.train_data, self.val_data, self.test_data = splitter.split(
            user_indices=self.processed_data['user_indices'],
            item_indices=self.processed_data['book_indices'],
            ratings=self.processed_data['ratings']
        )
        
        print(f"\n数据集划分完成:")
        print(f"  训练集: {len(self.train_data['user_indices'])} 条交互")
        print(f"  验证集: {len(self.val_data['user_indices'])} 条交互")
        print(f"  测试集: {len(self.test_data['user_indices'])} 条交互")
    
    def create_model(self):
        """创建模型"""
        print("\n" + "="*50)
        print("步骤4: 模型创建")
        print("="*50)
        
        user_feature_dim = self.processed_data['user_features'].shape[1]
        book_feature_dim = self.processed_data['book_features'].shape[1]
        
        self.model = DualNetworkGNN(
            user_feature_dim=user_feature_dim,
            book_feature_dim=book_feature_dim,
            embed_dim=self.config['embed_dim'],
            ub_gnn_layers=self.config['ub_gnn_layers'],
            bb_gnn_layers=self.config['bb_gnn_layers'],
            ub_gnn_type=self.config['ub_gnn_type'],
            bb_gnn_type=self.config['bb_gnn_type'],
            fusion_type=self.config['fusion_type'],
            prediction_type=self.config['prediction_type'],
            dropout=self.config['dropout']
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n模型创建完成:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def train_model(self):
        """训练模型"""
        print("\n" + "="*50)
        print("步骤5: 模型训练")
        print("="*50)
        
        self.trainer = Trainer(
            model=self.model,
            device=self.device,
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            batch_size=self.config['batch_size'],
            num_negatives=self.config['num_negatives'],
            loss_type=self.config['loss_type'],
            explicit_weight=self.config['explicit_weight'],
            implicit_weight=self.config['implicit_weight']
        )
        
        self.training_history = self.trainer.train(
            ub_data=self.ub_graph,
            bb_data=self.bb_graph,
            train_data=self.train_data,
            val_data=self.val_data,
            num_epochs=self.config['num_epochs'],
            early_stopping_patience=self.config['early_stopping_patience'],
            verbose=True
        )
        
        print(f"\n训练完成:")
        print(f"  最佳验证损失: {self.training_history['best_val_loss']:.4f}")
        print(f"  最佳轮次: {self.training_history['best_epoch'] + 1}")
        print(f"  总训练轮次: {len(self.training_history['train_losses'])}")
    
    def evaluate_model(self):
        """评估模型"""
        print("\n" + "="*50)
        print("步骤6: 模型评估")
        print("="*50)
        
        self.evaluator = ModelEvaluator(
            model=self.model,
            device=self.device,
            k_values=self.config['k_values'],
            batch_size=self.config['batch_size']
        )
        
        self.evaluation_results = self.evaluator.evaluate_on_test_set(
            ub_data=self.ub_graph,
            bb_data=self.bb_graph,
            test_data=self.test_data,
            k=max(self.config['k_values']),
            item_features=self.processed_data['book_features'].numpy()
        )
        
        print(f"\n评估结果:")
        for metric, value in self.evaluation_results.items():
            print(f"  {metric}: {value:.4f}")
    
    def visualize_results(self, save_path: Optional[str] = None):
        """可视化结果"""
        print("\n" + "="*50)
        print("步骤7: 结果可视化")
        print("="*50)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('双重网络GNN推荐系统结果', fontsize=16, fontweight='bold')
        
        # 1. 训练损失曲线
        ax1 = axes[0, 0]
        epochs = range(1, len(self.training_history['train_losses']) + 1)
        ax1.plot(epochs, self.training_history['train_losses'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, self.training_history['val_losses'], 'r-', label='验证损失', linewidth=2)
        ax1.axvline(x=self.training_history['best_epoch'] + 1, color='g', linestyle='--', 
                   label=f'最佳轮次 ({self.training_history["best_epoch"] + 1})')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.set_title('训练过程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 排序指标
        ax2 = axes[0, 1]
        ranking_metrics = {}
        for k in self.config['k_values']:
            for metric in ['Precision', 'Recall', 'NDCG']:
                key = f'{metric}@{k}'
                if key in self.evaluation_results:
                    if metric not in ranking_metrics:
                        ranking_metrics[metric] = []
                    ranking_metrics[metric].append(self.evaluation_results[key])
        
        x = np.arange(len(self.config['k_values']))
        width = 0.25
        
        for i, (metric, values) in enumerate(ranking_metrics.items()):
            ax2.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax2.set_xlabel('K值')
        ax2.set_ylabel('得分')
        ax2.set_title('排序指标 (Top-K)')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([f'K={k}' for k in self.config['k_values']])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 其他指标
        ax3 = axes[1, 0]
        other_metrics = {}
        for key, value in self.evaluation_results.items():
            if '@' not in key and key not in ['RMSE', 'MAE']:
                other_metrics[key] = value
        
        if other_metrics:
            metrics = list(other_metrics.keys())
            values = list(other_metrics.values())
            
            bars = ax3.bar(metrics, values, alpha=0.8, color='skyblue')
            ax3.set_ylabel('得分')
            ax3.set_title('其他评估指标')
            ax3.tick_params(axis='x', rotation=45)
            
            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 4. 评分预测指标（如果有）
        ax4 = axes[1, 1]
        rating_metrics = {}
        for key in ['RMSE', 'MAE']:
            if key in self.evaluation_results:
                rating_metrics[key] = self.evaluation_results[key]
        
        if rating_metrics:
            metrics = list(rating_metrics.keys())
            values = list(rating_metrics.values())
            
            bars = ax4.bar(metrics, values, alpha=0.8, color='lightcoral')
            ax4.set_ylabel('误差')
            ax4.set_title('评分预测指标')
            
            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, '无评分预测数据', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('评分预测指标')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果图保存到: {save_path}")
        
        plt.show()
    
    def save_results(self, save_dir: str):
        """保存结果"""
        print("\n" + "="*50)
        print("步骤8: 保存结果")
        print("="*50)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python类型
            history_to_save = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    history_to_save[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
                else:
                    history_to_save[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
            json.dump(history_to_save, f, indent=2)
        
        # 保存评估结果
        results_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            results_to_save = {k: float(v) for k, v in self.evaluation_results.items()}
            json.dump(results_to_save, f, indent=2)
        
        # 保存模型
        model_path = os.path.join(save_dir, 'model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'processed_data_info': {
                'num_users': self.processed_data['num_users'],
                'num_books': self.processed_data['num_books'],
                'user_feature_dim': self.processed_data['user_features'].shape[1],
                'book_feature_dim': self.processed_data['book_features'].shape[1]
            }
        }, model_path)
        
        # 保存数据映射
        mappings_path = os.path.join(save_dir, 'id_mappings.json')
        with open(mappings_path, 'w', encoding='utf-8') as f:
            mappings = {
                'user_id_map': {str(k): v for k, v in self.processed_data['user_id_map'].items()},
                'book_id_map': {str(k): v for k, v in self.processed_data['book_id_map'].items()},
                'reverse_user_map': {str(k): v for k, v in self.processed_data['reverse_user_map'].items()},
                'reverse_book_map': {str(k): v for k, v in self.processed_data['reverse_book_map'].items()}
            }
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        
        print(f"结果保存到: {save_dir}")
        print(f"  配置文件: {config_path}")
        print(f"  训练历史: {history_path}")
        print(f"  评估结果: {results_path}")
        print(f"  模型文件: {model_path}")
        print(f"  ID映射: {mappings_path}")
    
    def run_full_pipeline(self, save_dir: Optional[str] = None):
        """运行完整的推荐系统流程"""
        print("开始运行双重网络GNN书籍推荐系统")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行各个步骤
        self.load_and_preprocess_data()
        self.build_graphs()
        self.split_data()
        self.create_model()
        self.train_model()
        self.evaluate_model()
        
        # 可视化结果
        if save_dir:
            plot_path = os.path.join(save_dir, 'results_visualization.png')
            self.visualize_results(plot_path)
        else:
            self.visualize_results()
        
        # 保存结果
        if save_dir:
            self.save_results(save_dir)
        
        print("\n" + "="*50)
        print("推荐系统运行完成!")
        print("="*50)
        
        return self.evaluation_results


def get_default_config() -> Dict:
    """获取默认配置"""
    return {
        # 数据相关
        'data_path': 'data/books/',
        'min_user_interactions': 5,
        'min_book_interactions': 5,
        
        # 图构建相关
        'ub_feedback_handling': 'separate',  # 'separate', 'unified', 'explicit_only'
        'bb_similarity_method': 'content',   # 'content', 'collaborative', 'hybrid'
        'bb_similarity_threshold': 0.1,
        'bb_max_connections': 50,
        
        # 数据划分相关
        'split_method': 'random',  # 'random', 'user_based'
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'random_state': 42,
        
        # 模型相关
        'embed_dim': 64,
        'ub_gnn_layers': 2,
        'bb_gnn_layers': 2,
        'ub_gnn_type': 'lightgcn',  # 'lightgcn', 'sage', 'gat', 'gcn'
        'bb_gnn_type': 'sage',      # 'sage', 'gat', 'gcn'
        'fusion_type': 'attention', # 'concat', 'attention', 'gate', 'average'
        'prediction_type': 'dot_product',  # 'dot_product', 'mlp'
        'dropout': 0.1,
        
        # 训练相关
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 1024,
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'num_negatives': 1,
        'loss_type': 'multitask',  # 'bpr', 'mse', 'multitask'
        'explicit_weight': 0.5,
        'implicit_weight': 0.5,
        
        # 评估相关
        'k_values': [5, 10, 20]
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双重网络GNN书籍推荐系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--data_path', type=str, default='data/books/', help='数据路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # 更新数据路径
    if args.data_path:
        config['data_path'] = args.data_path
    
    # 创建保存目录
    if args.save_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.save_dir, f'dual_gnn_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    
    # 创建推荐系统并运行
    system = DualNetworkRecommendationSystem(config)
    results = system.run_full_pipeline(save_dir)
    
    # 打印最终结果摘要
    print(f"\n最终评估结果摘要:")
    for metric in ['Precision@10', 'Recall@10', 'NDCG@10', 'MRR', 'Coverage', 'Diversity']:
        if metric in results:
            print(f"  {metric}: {results[metric]:.4f}")


if __name__ == "__main__":
    main()