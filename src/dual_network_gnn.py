"""
双重网络GNN模型
实现用户-书籍交互图和书籍-书籍相似图的双重网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, Union, List
import numpy as np


class LightGCNConv(nn.Module):
    """LightGCN卷积层实现"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        LightGCN前向传播
        
        Args:
            x: 节点特征 [num_nodes, feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
        
        Returns:
            更新后的节点特征
        """
        from torch_geometric.utils import degree
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # 归一化
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # 消息传递
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.view(-1, 1))
        
        return out


class UserBookGNN(nn.Module):
    """用户-书籍交互图GNN编码器"""
    
    def __init__(self, 
                 user_feature_dim: int,
                 book_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 gnn_type: str = 'lightgcn',
                 dropout: float = 0.1):
        """
        初始化用户-书籍GNN
        
        Args:
            user_feature_dim: 用户特征维度
            book_feature_dim: 书籍特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            gnn_type: GNN类型 ('lightgcn', 'sage', 'gat', 'gcn')
            dropout: Dropout率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # 特征投影层
        self.user_proj = Linear(user_feature_dim, hidden_dim)
        self.book_proj = Linear(book_feature_dim, hidden_dim)
        
        # GNN层
        self.convs = nn.ModuleList()
        
        if gnn_type == 'lightgcn':
            # LightGCN不需要额外的卷积层，使用简单的消息传递
            for _ in range(num_layers):
                self.convs.append(LightGCNConv())
        else:
            # 构建异构卷积层
            for i in range(num_layers):
                conv_dict = {}
                
                if gnn_type == 'sage':
                    conv_dict[('user', 'rates', 'book')] = SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr='mean'
                    )
                    conv_dict[('user', 'implicitly_likes', 'book')] = SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr='mean'
                    )
                    conv_dict[('book', 'rev_rates', 'user')] = SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr='mean'
                    )
                    conv_dict[('book', 'rev_implicitly_likes', 'user')] = SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr='mean'
                    )
                elif gnn_type == 'gat':
                    conv_dict[('user', 'rates', 'book')] = GATConv(
                        (hidden_dim, hidden_dim), hidden_dim // 4, heads=4, dropout=dropout
                    )
                    conv_dict[('user', 'implicitly_likes', 'book')] = GATConv(
                        (hidden_dim, hidden_dim), hidden_dim // 4, heads=4, dropout=dropout
                    )
                    conv_dict[('book', 'rev_rates', 'user')] = GATConv(
                        (hidden_dim, hidden_dim), hidden_dim // 4, heads=4, dropout=dropout
                    )
                    conv_dict[('book', 'rev_implicitly_likes', 'user')] = GATConv(
                        (hidden_dim, hidden_dim), hidden_dim // 4, heads=4, dropout=dropout
                    )
                elif gnn_type == 'gcn':
                    conv_dict[('user', 'rates', 'book')] = GCNConv(hidden_dim, hidden_dim)
                    conv_dict[('user', 'implicitly_likes', 'book')] = GCNConv(hidden_dim, hidden_dim)
                    conv_dict[('book', 'rev_rates', 'user')] = GCNConv(hidden_dim, hidden_dim)
                    conv_dict[('book', 'rev_implicitly_likes', 'user')] = GCNConv(hidden_dim, hidden_dim)
                
                self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: 异构图数据
        
        Returns:
            (用户嵌入, 书籍嵌入)
        """
        # 特征投影
        x_dict = {
            'user': self.user_proj(data['user'].x),
            'book': self.book_proj(data['book'].x)
        }
        
        if self.gnn_type == 'lightgcn':
            return self._lightgcn_forward(data, x_dict)
        else:
            return self._hetero_forward(data, x_dict)
    
    def _lightgcn_forward(self, data: HeteroData, x_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """LightGCN前向传播"""
        # 收集所有层的嵌入
        user_embeds = [x_dict['user']]
        book_embeds = [x_dict['book']]
        
        current_user_embed = x_dict['user']
        current_book_embed = x_dict['book']
        
        for conv in self.convs:
            # 构建二分图的边索引
            edge_indices = []
            
            # 收集所有用户-书籍边
            if ('user', 'rates', 'book') in data.edge_types:
                rates_edges = data['user', 'rates', 'book'].edge_index
                edge_indices.append(rates_edges)
            
            if ('user', 'implicitly_likes', 'book') in data.edge_types:
                implicit_edges = data['user', 'implicitly_likes', 'book'].edge_index
                edge_indices.append(implicit_edges)
            
            if edge_indices:
                # 合并所有边
                all_edges = torch.cat(edge_indices, dim=1)
                
                # 创建双向边（用户->书籍 和 书籍->用户）
                user_to_book = all_edges
                book_to_user = torch.stack([
                    all_edges[1] + len(current_user_embed),  # 书籍索引偏移
                    all_edges[0]  # 用户索引
                ])
                
                # 合并用户和书籍嵌入
                combined_embed = torch.cat([current_user_embed, current_book_embed], dim=0)
                
                # 创建完整的边索引
                full_edge_index = torch.cat([user_to_book, book_to_user], dim=1)
                
                # LightGCN传播
                updated_embed = conv(combined_embed, full_edge_index)
                
                # 分离用户和书籍嵌入
                num_users = current_user_embed.size(0)
                current_user_embed = updated_embed[:num_users]
                current_book_embed = updated_embed[num_users:]
            
            user_embeds.append(current_user_embed)
            book_embeds.append(current_book_embed)
        
        # 层聚合（平均）
        final_user_embed = torch.stack(user_embeds).mean(dim=0)
        final_book_embed = torch.stack(book_embeds).mean(dim=0)
        
        return final_user_embed, final_book_embed
    
    def _hetero_forward(self, data: HeteroData, x_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """异构GNN前向传播"""
        # 添加反向边以支持双向消息传递
        edge_index_dict = {}
        
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index
            
            # 添加反向边
            src, rel, dst = edge_type
            if src != dst:  # 避免自环的反向边
                rev_edge_type = (dst, f'rev_{rel}', src)
                edge_index_dict[rev_edge_type] = torch.stack([
                    data[edge_type].edge_index[1],
                    data[edge_type].edge_index[0]
                ])
        
        # GNN层传播
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
            # 应用激活函数和Dropout
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = self.dropout_layer(x_dict[node_type])
        
        return x_dict['user'], x_dict['book']


class BookBookGNN(nn.Module):
    """书籍-书籍相似图GNN编码器"""
    
    def __init__(self,
                 book_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 gnn_type: str = 'sage',
                 dropout: float = 0.1):
        """
        初始化书籍-书籍GNN
        
        Args:
            book_feature_dim: 书籍特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            gnn_type: GNN类型 ('sage', 'gat', 'gcn')
            dropout: Dropout率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # 特征投影层
        self.book_proj = Linear(book_feature_dim, hidden_dim)
        
        # GNN层
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 异构图数据
        
        Returns:
            书籍嵌入
        """
        # 特征投影
        x = self.book_proj(data['book'].x)
        
        # 获取边索引和权重
        edge_index = data['book', 'similar_to', 'book'].edge_index
        edge_weight = data['book', 'similar_to', 'book'].get('edge_weight', None)
        
        # GNN层传播
        for conv in self.convs:
            if self.gnn_type == 'gcn' and edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        return x


class EmbeddingFusion(nn.Module):
    """嵌入融合模块"""
    
    def __init__(self,
                 embed_dim: int,
                 fusion_type: str = 'attention',
                 dropout: float = 0.1):
        """
        初始化嵌入融合模块
        
        Args:
            embed_dim: 嵌入维度
            fusion_type: 融合类型 ('concat', 'attention', 'gate', 'average')
            dropout: Dropout率
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        
        if fusion_type == 'concat':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
        elif fusion_type == 'attention':
            self.attention_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == 'gate':
            self.gate_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
        elif fusion_type != 'average':
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def forward(self, 
                collab_embed: torch.Tensor, 
                content_embed: torch.Tensor) -> torch.Tensor:
        """
        融合协同嵌入和内容嵌入
        
        Args:
            collab_embed: 协同过滤嵌入
            content_embed: 内容嵌入
        
        Returns:
            融合后的嵌入
        """
        if self.fusion_type == 'concat':
            combined = torch.cat([collab_embed, content_embed], dim=-1)
            return self.fusion_mlp(combined)
        
        elif self.fusion_type == 'attention':
            combined = torch.cat([collab_embed, content_embed], dim=-1)
            attention_weights = self.attention_mlp(combined)
            
            weighted_collab = attention_weights[:, 0:1] * collab_embed
            weighted_content = attention_weights[:, 1:2] * content_embed
            
            return weighted_collab + weighted_content
        
        elif self.fusion_type == 'gate':
            combined = torch.cat([collab_embed, content_embed], dim=-1)
            gate = self.gate_mlp(combined)
            
            return gate * collab_embed + (1 - gate) * content_embed
        
        elif self.fusion_type == 'average':
            return (collab_embed + content_embed) / 2
        
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")


class PredictionLayer(nn.Module):
    """预测层"""
    
    def __init__(self,
                 embed_dim: int,
                 prediction_type: str = 'dot_product',
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        """
        初始化预测层
        
        Args:
            embed_dim: 嵌入维度
            prediction_type: 预测类型 ('dot_product', 'mlp')
            hidden_dim: MLP隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.prediction_type = prediction_type
        
        if prediction_type == 'mlp':
            if hidden_dim is None:
                hidden_dim = embed_dim
            
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif prediction_type != 'dot_product':
            raise ValueError(f"不支持的预测类型: {prediction_type}")
    
    def forward(self, 
                user_embed: torch.Tensor, 
                book_embed: torch.Tensor) -> torch.Tensor:
        """
        预测用户对书籍的偏好
        
        Args:
            user_embed: 用户嵌入
            book_embed: 书籍嵌入
        
        Returns:
            预测得分
        """
        if self.prediction_type == 'dot_product':
            return (user_embed * book_embed).sum(dim=-1)
        
        elif self.prediction_type == 'mlp':
            combined = torch.cat([user_embed, book_embed], dim=-1)
            return self.mlp(combined).squeeze(-1)
        
        else:
            raise ValueError(f"不支持的预测类型: {self.prediction_type}")


class DualNetworkGNN(nn.Module):
    """双重网络GNN推荐模型"""
    
    def __init__(self,
                 user_feature_dim: int,
                 book_feature_dim: int,
                 embed_dim: int = 64,
                 ub_gnn_layers: int = 2,
                 bb_gnn_layers: int = 2,
                 ub_gnn_type: str = 'lightgcn',
                 bb_gnn_type: str = 'sage',
                 fusion_type: str = 'attention',
                 prediction_type: str = 'dot_product',
                 dropout: float = 0.1):
        """
        初始化双重网络GNN模型
        
        Args:
            user_feature_dim: 用户特征维度
            book_feature_dim: 书籍特征维度
            embed_dim: 嵌入维度
            ub_gnn_layers: 用户-书籍GNN层数
            bb_gnn_layers: 书籍-书籍GNN层数
            ub_gnn_type: 用户-书籍GNN类型
            bb_gnn_type: 书籍-书籍GNN类型
            fusion_type: 融合类型
            prediction_type: 预测类型
            dropout: Dropout率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 用户-书籍交互图GNN
        self.ub_gnn = UserBookGNN(
            user_feature_dim=user_feature_dim,
            book_feature_dim=book_feature_dim,
            hidden_dim=embed_dim,
            num_layers=ub_gnn_layers,
            gnn_type=ub_gnn_type,
            dropout=dropout
        )
        
        # 书籍-书籍相似图GNN
        self.bb_gnn = BookBookGNN(
            book_feature_dim=book_feature_dim,
            hidden_dim=embed_dim,
            num_layers=bb_gnn_layers,
            gnn_type=bb_gnn_type,
            dropout=dropout
        )
        
        # 嵌入融合模块
        self.fusion = EmbeddingFusion(
            embed_dim=embed_dim,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # 预测层
        self.predictor = PredictionLayer(
            embed_dim=embed_dim,
            prediction_type=prediction_type,
            dropout=dropout
        )
    
    def forward(self, 
                ub_data: HeteroData, 
                bb_data: HeteroData,
                user_indices: Optional[torch.Tensor] = None,
                book_indices: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        前向传播
        
        Args:
            ub_data: 用户-书籍交互图数据
            bb_data: 书籍-书籍相似图数据
            user_indices: 用户索引（用于预测）
            book_indices: 书籍索引（用于预测）
        
        Returns:
            如果提供了user_indices和book_indices，返回预测得分
            否则返回(用户嵌入, 融合后的书籍嵌入)
        """
        # 从用户-书籍交互图获取嵌入
        user_embed_collab, book_embed_collab = self.ub_gnn(ub_data)
        
        # 从书籍-书籍相似图获取嵌入
        book_embed_content = self.bb_gnn(bb_data)
        
        # 融合书籍嵌入
        book_embed_final = self.fusion(book_embed_collab, book_embed_content)
        
        # 如果提供了索引，进行预测
        if user_indices is not None and book_indices is not None:
            user_embed_selected = user_embed_collab[user_indices]
            book_embed_selected = book_embed_final[book_indices]
            
            predictions = self.predictor(user_embed_selected, book_embed_selected)
            return predictions
        
        # 否则返回嵌入
        return user_embed_collab, book_embed_final
    
    def predict(self, 
                ub_data: HeteroData, 
                bb_data: HeteroData,
                user_indices: torch.Tensor,
                book_indices: torch.Tensor) -> torch.Tensor:
        """
        预测用户对书籍的偏好
        
        Args:
            ub_data: 用户-书籍交互图数据
            bb_data: 书籍-书籍相似图数据
            user_indices: 用户索引
            book_indices: 书籍索引
        
        Returns:
            预测得分
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(ub_data, bb_data, user_indices, book_indices)
        return predictions
    
    def get_embeddings(self, 
                      ub_data: HeteroData, 
                      bb_data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取用户和书籍的最终嵌入
        
        Args:
            ub_data: 用户-书籍交互图数据
            bb_data: 书籍-书籍相似图数据
        
        Returns:
            (用户嵌入, 书籍嵌入)
        """
        self.eval()
        with torch.no_grad():
            user_embed, book_embed = self.forward(ub_data, bb_data)
        return user_embed, book_embed


if __name__ == "__main__":
    # 测试模型
    print("测试双重网络GNN模型...")
    
    # 创建示例数据
    num_users, num_books = 100, 200
    user_feature_dim, book_feature_dim = 10, 50
    
    # 创建示例异构图数据
    ub_data = HeteroData()
    ub_data['user'].x = torch.randn(num_users, user_feature_dim)
    ub_data['book'].x = torch.randn(num_books, book_feature_dim)
    ub_data['user', 'rates', 'book'].edge_index = torch.randint(0, min(num_users, num_books), (2, 500))
    
    bb_data = HeteroData()
    bb_data['book'].x = torch.randn(num_books, book_feature_dim)
    bb_data['book', 'similar_to', 'book'].edge_index = torch.randint(0, num_books, (2, 1000))
    bb_data['book', 'similar_to', 'book'].edge_weight = torch.rand(1000)
    
    # 创建模型
    model = DualNetworkGNN(
        user_feature_dim=user_feature_dim,
        book_feature_dim=book_feature_dim,
        embed_dim=64
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    user_embed, book_embed = model(ub_data, bb_data)
    print(f"用户嵌入形状: {user_embed.shape}")
    print(f"书籍嵌入形状: {book_embed.shape}")
    
    # 测试预测
    user_indices = torch.randint(0, num_users, (50,))
    book_indices = torch.randint(0, num_books, (50,))
    predictions = model(ub_data, bb_data, user_indices, book_indices)
    print(f"预测得分形状: {predictions.shape}")
    
    print("模型测试完成!")