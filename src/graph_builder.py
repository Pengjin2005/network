"""
图构建模块
构建用户-书籍交互图和书籍-书籍相似图
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Tuple, List, Optional
import pandas as pd
from scipy.sparse import coo_matrix
import warnings

warnings.filterwarnings('ignore')


class GraphBuilder:
    """双重网络图构建器"""
    
    def __init__(self, processed_data: Dict):
        """
        初始化图构建器
        
        Args:
            processed_data: 预处理后的数据字典
        """
        self.processed_data = processed_data
        self.num_users = processed_data['num_users']
        self.num_books = processed_data['num_books']
        
        # 提取数据
        self.user_features = processed_data['user_features']
        self.book_features = processed_data['book_features']
        self.user_indices = processed_data['user_indices']
        self.book_indices = processed_data['book_indices']
        self.ratings = processed_data['ratings']
        
    def build_user_book_interaction_graph(self, 
                                        handle_implicit_feedback: str = 'separate') -> HeteroData:
        """
        构建用户-书籍交互图
        
        Args:
            handle_implicit_feedback: 处理隐式反馈的方式
                - 'separate': 区分显式和隐式反馈
                - 'unified': 统一为隐式交互
                - 'explicit_only': 仅使用显式反馈
        
        Returns:
            HeteroData对象
        """
        print(f"构建用户-书籍交互图 (处理方式: {handle_implicit_feedback})...")
        
        data = HeteroData()
        
        # 添加节点特征
        data['user'].x = self.user_features
        data['book'].x = self.book_features
        
        # 根据处理方式构建边
        if handle_implicit_feedback == 'separate':
            self._build_separate_edges(data)
        elif handle_implicit_feedback == 'unified':
            self._build_unified_edges(data)
        elif handle_implicit_feedback == 'explicit_only':
            self._build_explicit_only_edges(data)
        else:
            raise ValueError(f"不支持的处理方式: {handle_implicit_feedback}")
        
        print(f"用户-书籍交互图构建完成")
        print(f"  用户节点: {data['user'].x.shape[0]}")
        print(f"  书籍节点: {data['book'].x.shape[0]}")
        
        return data
    
    def _build_separate_edges(self, data: HeteroData) -> None:
        """构建区分显式和隐式反馈的边"""
        # 显式评分 (1-10)
        explicit_mask = self.ratings > 0
        if np.any(explicit_mask):
            explicit_users = self.user_indices[explicit_mask]
            explicit_books = self.book_indices[explicit_mask]
            explicit_ratings = self.ratings[explicit_mask]
            
            edge_index = torch.stack([
                torch.LongTensor(explicit_users),
                torch.LongTensor(explicit_books)
            ])
            
            data['user', 'rates', 'book'].edge_index = edge_index
            data['user', 'rates', 'book'].edge_attr = torch.FloatTensor(explicit_ratings).unsqueeze(1)
            
            print(f"  显式评分边: {edge_index.shape[1]}")
        
        # 隐式反馈 (0分)
        implicit_mask = self.ratings == 0
        if np.any(implicit_mask):
            implicit_users = self.user_indices[implicit_mask]
            implicit_books = self.book_indices[implicit_mask]
            
            edge_index = torch.stack([
                torch.LongTensor(implicit_users),
                torch.LongTensor(implicit_books)
            ])
            
            data['user', 'implicitly_likes', 'book'].edge_index = edge_index
            
            print(f"  隐式反馈边: {edge_index.shape[1]}")
    
    def _build_unified_edges(self, data: HeteroData) -> None:
        """构建统一的交互边"""
        edge_index = torch.stack([
            torch.LongTensor(self.user_indices),
            torch.LongTensor(self.book_indices)
        ])
        
        data['user', 'interacts', 'book'].edge_index = edge_index
        
        print(f"  统一交互边: {edge_index.shape[1]}")
    
    def _build_explicit_only_edges(self, data: HeteroData) -> None:
        """构建仅显式反馈的边"""
        explicit_mask = self.ratings > 0
        if np.any(explicit_mask):
            explicit_users = self.user_indices[explicit_mask]
            explicit_books = self.book_indices[explicit_mask]
            explicit_ratings = self.ratings[explicit_mask]
            
            edge_index = torch.stack([
                torch.LongTensor(explicit_users),
                torch.LongTensor(explicit_books)
            ])
            
            data['user', 'rates', 'book'].edge_index = edge_index
            data['user', 'rates', 'book'].edge_attr = torch.FloatTensor(explicit_ratings).unsqueeze(1)
            
            print(f"  显式评分边: {edge_index.shape[1]}")
    
    def build_book_similarity_graph(self, 
                                  similarity_method: str = 'content',
                                  similarity_threshold: float = 0.1,
                                  max_connections: int = 50) -> HeteroData:
        """
        构建书籍-书籍相似图
        
        Args:
            similarity_method: 相似性计算方法
                - 'content': 基于内容特征
                - 'collaborative': 基于协同信号
                - 'hybrid': 混合方法
            similarity_threshold: 相似性阈值
            max_connections: 每个节点的最大连接数
        
        Returns:
            HeteroData对象
        """
        print(f"构建书籍-书籍相似图 (方法: {similarity_method})...")
        
        data = HeteroData()
        
        # 添加节点特征
        data['book'].x = self.book_features
        
        # 根据方法计算相似性
        if similarity_method == 'content':
            similarity_matrix = self._compute_content_similarity()
        elif similarity_method == 'collaborative':
            similarity_matrix = self._compute_collaborative_similarity()
        elif similarity_method == 'hybrid':
            content_sim = self._compute_content_similarity()
            collab_sim = self._compute_collaborative_similarity()
            similarity_matrix = 0.7 * content_sim + 0.3 * collab_sim
        else:
            raise ValueError(f"不支持的相似性方法: {similarity_method}")
        
        # 构建边
        edge_index, edge_weights = self._build_similarity_edges(
            similarity_matrix, similarity_threshold, max_connections
        )
        
        data['book', 'similar_to', 'book'].edge_index = edge_index
        data['book', 'similar_to', 'book'].edge_weight = edge_weights
        
        print(f"书籍-书籍相似图构建完成")
        print(f"  书籍节点: {data['book'].x.shape[0]}")
        print(f"  相似性边: {edge_index.shape[1]}")
        
        return data
    
    def _compute_content_similarity(self) -> np.ndarray:
        """计算基于内容的相似性"""
        print("计算内容相似性...")
        
        # 使用书籍特征计算余弦相似度
        book_features_np = self.book_features.numpy()
        similarity_matrix = cosine_similarity(book_features_np)
        
        # 将对角线设为0（自己与自己的相似性）
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _compute_collaborative_similarity(self) -> np.ndarray:
        """计算基于协同信号的相似性"""
        print("计算协同相似性...")
        
        # 构建用户-书籍交互矩阵
        interaction_matrix = coo_matrix(
            (np.ones(len(self.user_indices)), 
             (self.user_indices, self.book_indices)),
            shape=(self.num_users, self.num_books)
        ).tocsr()
        
        # 计算书籍-书籍相似性（基于共同用户）
        # 转置矩阵，使得行为书籍，列为用户
        book_user_matrix = interaction_matrix.T
        
        # 计算余弦相似度
        similarity_matrix = cosine_similarity(book_user_matrix)
        
        # 将对角线设为0
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _build_similarity_edges(self, 
                              similarity_matrix: np.ndarray,
                              threshold: float,
                              max_connections: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据相似性矩阵构建边"""
        print(f"构建相似性边 (阈值: {threshold}, 最大连接: {max_connections})...")
        
        edges = []
        weights = []
        
        for i in range(similarity_matrix.shape[0]):
            # 获取与书籍i相似的书籍
            similarities = similarity_matrix[i]
            
            # 过滤低于阈值的相似性
            valid_indices = np.where(similarities >= threshold)[0]
            valid_similarities = similarities[valid_indices]
            
            # 按相似性排序，取前max_connections个
            if len(valid_indices) > max_connections:
                top_indices = np.argsort(valid_similarities)[-max_connections:]
                valid_indices = valid_indices[top_indices]
                valid_similarities = valid_similarities[top_indices]
            
            # 添加边
            for j, sim in zip(valid_indices, valid_similarities):
                edges.append([i, j])
                weights.append(sim)
        
        if len(edges) == 0:
            # 如果没有边，创建空的边索引
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0, dtype=torch.float)
        else:
            edges = np.array(edges).T
            edge_index = torch.LongTensor(edges)
            edge_weights = torch.FloatTensor(weights)
        
        return edge_index, edge_weights
    
    def build_dual_graphs(self, 
                         ub_feedback_handling: str = 'separate',
                         bb_similarity_method: str = 'content',
                         bb_similarity_threshold: float = 0.1,
                         bb_max_connections: int = 50) -> Tuple[HeteroData, HeteroData]:
        """
        构建双重图
        
        Args:
            ub_feedback_handling: 用户-书籍图的反馈处理方式
            bb_similarity_method: 书籍-书籍图的相似性方法
            bb_similarity_threshold: 书籍相似性阈值
            bb_max_connections: 书籍最大连接数
        
        Returns:
            (用户-书籍图, 书籍-书籍图)
        """
        print("构建双重图...")
        
        # 构建用户-书籍交互图
        ub_graph = self.build_user_book_interaction_graph(ub_feedback_handling)
        
        # 构建书籍-书籍相似图
        bb_graph = self.build_book_similarity_graph(
            bb_similarity_method, bb_similarity_threshold, bb_max_connections
        )
        
        print("双重图构建完成!")
        return ub_graph, bb_graph
    
    def create_unified_heterograph(self,
                                 ub_feedback_handling: str = 'separate',
                                 bb_similarity_method: str = 'content',
                                 bb_similarity_threshold: float = 0.1,
                                 bb_max_connections: int = 50) -> HeteroData:
        """
        创建统一的异构图（包含所有节点类型和边类型）
        
        Returns:
            统一的HeteroData对象
        """
        print("创建统一异构图...")
        
        data = HeteroData()
        
        # 添加节点特征
        data['user'].x = self.user_features
        data['book'].x = self.book_features
        
        # 添加用户-书籍交互边
        if ub_feedback_handling == 'separate':
            self._build_separate_edges(data)
        elif ub_feedback_handling == 'unified':
            self._build_unified_edges(data)
        elif ub_feedback_handling == 'explicit_only':
            self._build_explicit_only_edges(data)
        
        # 添加书籍-书籍相似边
        if bb_similarity_method == 'content':
            similarity_matrix = self._compute_content_similarity()
        elif bb_similarity_method == 'collaborative':
            similarity_matrix = self._compute_collaborative_similarity()
        elif bb_similarity_method == 'hybrid':
            content_sim = self._compute_content_similarity()
            collab_sim = self._compute_collaborative_similarity()
            similarity_matrix = 0.7 * content_sim + 0.3 * collab_sim
        
        edge_index, edge_weights = self._build_similarity_edges(
            similarity_matrix, bb_similarity_threshold, bb_max_connections
        )
        
        data['book', 'similar_to', 'book'].edge_index = edge_index
        data['book', 'similar_to', 'book'].edge_weight = edge_weights
        
        print("统一异构图创建完成!")
        return data


def build_graphs(processed_data: Dict,
                ub_feedback_handling: str = 'separate',
                bb_similarity_method: str = 'content',
                bb_similarity_threshold: float = 0.1,
                bb_max_connections: int = 50,
                return_unified: bool = False) -> Tuple[HeteroData, ...]:
    """
    构建图的主函数
    
    Args:
        processed_data: 预处理后的数据
        ub_feedback_handling: 用户-书籍图的反馈处理方式
        bb_similarity_method: 书籍-书籍图的相似性方法
        bb_similarity_threshold: 书籍相似性阈值
        bb_max_connections: 书籍最大连接数
        return_unified: 是否返回统一的异构图
    
    Returns:
        如果return_unified=False: (用户-书籍图, 书籍-书籍图)
        如果return_unified=True: (统一异构图,)
    """
    builder = GraphBuilder(processed_data)
    
    if return_unified:
        unified_graph = builder.create_unified_heterograph(
            ub_feedback_handling, bb_similarity_method, 
            bb_similarity_threshold, bb_max_connections
        )
        return (unified_graph,)
    else:
        ub_graph, bb_graph = builder.build_dual_graphs(
            ub_feedback_handling, bb_similarity_method,
            bb_similarity_threshold, bb_max_connections
        )
        return ub_graph, bb_graph


if __name__ == "__main__":
    # 测试图构建
    from data_preprocessing import preprocess_book_crossing_data
    
    print("测试图构建...")
    
    # 预处理数据
    processed_data = preprocess_book_crossing_data()
    
    # 构建双重图
    ub_graph, bb_graph = build_graphs(processed_data)
    
    print(f"\n用户-书籍图:")
    print(f"  节点类型: {list(ub_graph.node_types)}")
    print(f"  边类型: {list(ub_graph.edge_types)}")
    
    print(f"\n书籍-书籍图:")
    print(f"  节点类型: {list(bb_graph.node_types)}")
    print(f"  边类型: {list(bb_graph.edge_types)}")
    
    # 构建统一异构图
    unified_graph = build_graphs(processed_data, return_unified=True)[0]
    
    print(f"\n统一异构图:")
    print(f"  节点类型: {list(unified_graph.node_types)}")
    print(f"  边类型: {list(unified_graph.edge_types)}")