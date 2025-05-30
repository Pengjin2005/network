"""
评估模块
包含各种推荐系统评估指标的计算
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class RecommendationEvaluator:
    """推荐系统评估器"""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        初始化评估器
        
        Args:
            k_values: Top-K评估的K值列表
        """
        self.k_values = k_values
    
    def precision_at_k(self, 
                      recommended_items: List[List[int]], 
                      relevant_items: List[List[int]], 
                      k: int) -> float:
        """
        计算Precision@K
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            relevant_items: 相关物品列表，每个用户一个列表
            k: K值
        
        Returns:
            Precision@K
        """
        precisions = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            if len(rec_items) == 0:
                precisions.append(0.0)
                continue
            
            # 取前K个推荐
            top_k_rec = rec_items[:k]
            rel_set = set(rel_items)
            
            # 计算命中数
            hits = len([item for item in top_k_rec if item in rel_set])
            precision = hits / min(len(top_k_rec), k)
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def recall_at_k(self, 
                   recommended_items: List[List[int]], 
                   relevant_items: List[List[int]], 
                   k: int) -> float:
        """
        计算Recall@K
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            relevant_items: 相关物品列表，每个用户一个列表
            k: K值
        
        Returns:
            Recall@K
        """
        recalls = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            if len(rel_items) == 0:
                recalls.append(0.0)
                continue
            
            # 取前K个推荐
            top_k_rec = rec_items[:k]
            rel_set = set(rel_items)
            
            # 计算命中数
            hits = len([item for item in top_k_rec if item in rel_set])
            recall = hits / len(rel_items)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def ndcg_at_k(self, 
                 recommended_items: List[List[int]], 
                 relevant_items: List[List[int]], 
                 k: int,
                 relevance_scores: Optional[List[List[float]]] = None) -> float:
        """
        计算NDCG@K
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            relevant_items: 相关物品列表，每个用户一个列表
            k: K值
            relevance_scores: 相关性得分，如果为None则使用二元相关性
        
        Returns:
            NDCG@K
        """
        ndcgs = []
        
        for i, (rec_items, rel_items) in enumerate(zip(recommended_items, relevant_items)):
            if len(rel_items) == 0:
                ndcgs.append(0.0)
                continue
            
            # 取前K个推荐
            top_k_rec = rec_items[:k]
            rel_set = set(rel_items)
            
            # 计算DCG
            dcg = 0.0
            for j, item in enumerate(top_k_rec):
                if item in rel_set:
                    if relevance_scores is not None and i < len(relevance_scores):
                        # 使用提供的相关性得分
                        rel_score = relevance_scores[i].get(item, 0.0) if isinstance(relevance_scores[i], dict) else 1.0
                    else:
                        # 使用二元相关性
                        rel_score = 1.0
                    
                    dcg += rel_score / np.log2(j + 2)  # j+2 因为log2(1)=0
            
            # 计算IDCG
            if relevance_scores is not None and i < len(relevance_scores):
                if isinstance(relevance_scores[i], dict):
                    ideal_scores = sorted([relevance_scores[i].get(item, 0.0) for item in rel_items], reverse=True)
                else:
                    ideal_scores = [1.0] * len(rel_items)
            else:
                ideal_scores = [1.0] * len(rel_items)
            
            idcg = 0.0
            for j, score in enumerate(ideal_scores[:k]):
                if score > 0:
                    idcg += score / np.log2(j + 2)
            
            # 计算NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs)
    
    def map_at_k(self, 
                recommended_items: List[List[int]], 
                relevant_items: List[List[int]], 
                k: int) -> float:
        """
        计算MAP@K (Mean Average Precision)
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            relevant_items: 相关物品列表，每个用户一个列表
            k: K值
        
        Returns:
            MAP@K
        """
        aps = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            if len(rel_items) == 0:
                aps.append(0.0)
                continue
            
            # 取前K个推荐
            top_k_rec = rec_items[:k]
            rel_set = set(rel_items)
            
            # 计算AP
            hits = 0
            precision_sum = 0.0
            
            for i, item in enumerate(top_k_rec):
                if item in rel_set:
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            if hits > 0:
                ap = precision_sum / min(len(rel_items), k)
            else:
                ap = 0.0
            
            aps.append(ap)
        
        return np.mean(aps)
    
    def mrr(self, 
           recommended_items: List[List[int]], 
           relevant_items: List[List[int]]) -> float:
        """
        计算MRR (Mean Reciprocal Rank)
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            relevant_items: 相关物品列表，每个用户一个列表
        
        Returns:
            MRR
        """
        rrs = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            if len(rel_items) == 0:
                rrs.append(0.0)
                continue
            
            rel_set = set(rel_items)
            
            # 找到第一个相关物品的排名
            for i, item in enumerate(rec_items):
                if item in rel_set:
                    rrs.append(1.0 / (i + 1))
                    break
            else:
                rrs.append(0.0)
        
        return np.mean(rrs)
    
    def coverage(self, 
                recommended_items: List[List[int]], 
                total_items: int) -> float:
        """
        计算覆盖率
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            total_items: 总物品数
        
        Returns:
            覆盖率
        """
        recommended_set = set()
        for rec_items in recommended_items:
            recommended_set.update(rec_items)
        
        return len(recommended_set) / total_items
    
    def diversity(self, 
                 recommended_items: List[List[int]], 
                 item_features: Optional[np.ndarray] = None,
                 similarity_matrix: Optional[np.ndarray] = None) -> float:
        """
        计算多样性
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            item_features: 物品特征矩阵
            similarity_matrix: 物品相似性矩阵
        
        Returns:
            平均多样性
        """
        if similarity_matrix is None and item_features is None:
            # 如果没有相似性信息，返回基于类别的多样性
            diversities = []
            for rec_items in recommended_items:
                if len(rec_items) <= 1:
                    diversities.append(0.0)
                else:
                    # 简单的基于唯一性的多样性
                    diversity = len(set(rec_items)) / len(rec_items)
                    diversities.append(diversity)
            return np.mean(diversities)
        
        diversities = []
        
        for rec_items in recommended_items:
            if len(rec_items) <= 1:
                diversities.append(0.0)
                continue
            
            # 计算推荐列表内物品的平均相似性
            similarities = []
            for i in range(len(rec_items)):
                for j in range(i + 1, len(rec_items)):
                    item_i, item_j = rec_items[i], rec_items[j]
                    
                    if similarity_matrix is not None:
                        if item_i < similarity_matrix.shape[0] and item_j < similarity_matrix.shape[1]:
                            sim = similarity_matrix[item_i, item_j]
                        else:
                            sim = 0.0
                    elif item_features is not None:
                        if item_i < item_features.shape[0] and item_j < item_features.shape[0]:
                            # 计算余弦相似度
                            feat_i = item_features[item_i]
                            feat_j = item_features[item_j]
                            norm_i = np.linalg.norm(feat_i)
                            norm_j = np.linalg.norm(feat_j)
                            if norm_i > 0 and norm_j > 0:
                                sim = np.dot(feat_i, feat_j) / (norm_i * norm_j)
                            else:
                                sim = 0.0
                        else:
                            sim = 0.0
                    else:
                        sim = 0.0
                    
                    similarities.append(sim)
            
            # 多样性 = 1 - 平均相似性
            if similarities:
                diversity = 1.0 - np.mean(similarities)
            else:
                diversity = 0.0
            
            diversities.append(diversity)
        
        return np.mean(diversities)
    
    def novelty(self, 
               recommended_items: List[List[int]], 
               item_popularity: np.ndarray) -> float:
        """
        计算新颖性
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            item_popularity: 物品流行度数组
        
        Returns:
            平均新颖性
        """
        novelties = []
        
        for rec_items in recommended_items:
            if len(rec_items) == 0:
                novelties.append(0.0)
                continue
            
            # 计算推荐列表的平均新颖性
            item_novelties = []
            for item in rec_items:
                if item < len(item_popularity):
                    # 新颖性 = -log(流行度)
                    popularity = item_popularity[item]
                    if popularity > 0:
                        novelty = -np.log2(popularity)
                    else:
                        novelty = 0.0
                else:
                    novelty = 0.0
                item_novelties.append(novelty)
            
            novelties.append(np.mean(item_novelties))
        
        return np.mean(novelties)
    
    def evaluate_ranking(self, 
                        recommended_items: List[List[int]], 
                        relevant_items: List[List[int]],
                        total_items: int,
                        item_features: Optional[np.ndarray] = None,
                        item_popularity: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        综合评估排序性能
        
        Args:
            recommended_items: 推荐物品列表，每个用户一个列表
            relevant_items: 相关物品列表，每个用户一个列表
            total_items: 总物品数
            item_features: 物品特征矩阵
            item_popularity: 物品流行度数组
        
        Returns:
            评估结果字典
        """
        results = {}
        
        # 排序指标
        for k in self.k_values:
            results[f'Precision@{k}'] = self.precision_at_k(recommended_items, relevant_items, k)
            results[f'Recall@{k}'] = self.recall_at_k(recommended_items, relevant_items, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(recommended_items, relevant_items, k)
            results[f'MAP@{k}'] = self.map_at_k(recommended_items, relevant_items, k)
        
        # MRR
        results['MRR'] = self.mrr(recommended_items, relevant_items)
        
        # 覆盖率
        results['Coverage'] = self.coverage(recommended_items, total_items)
        
        # 多样性
        results['Diversity'] = self.diversity(recommended_items, item_features)
        
        # 新颖性
        if item_popularity is not None:
            results['Novelty'] = self.novelty(recommended_items, item_popularity)
        
        return results
    
    def evaluate_rating_prediction(self, 
                                 predictions: np.ndarray, 
                                 targets: np.ndarray) -> Dict[str, float]:
        """
        评估评分预测性能
        
        Args:
            predictions: 预测评分
            targets: 真实评分
        
        Returns:
            评估结果字典
        """
        results = {}
        
        # RMSE
        results['RMSE'] = np.sqrt(mean_squared_error(targets, predictions))
        
        # MAE
        results['MAE'] = mean_absolute_error(targets, predictions)
        
        return results


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, 
                 model,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 k_values: List[int] = [5, 10, 20],
                 batch_size: int = 1024):
        """
        初始化模型评估器
        
        Args:
            model: 训练好的模型
            device: 设备
            k_values: Top-K评估的K值列表
            batch_size: 批大小
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.evaluator = RecommendationEvaluator(k_values)
    
    def generate_recommendations(self,
                               ub_data: HeteroData,
                               bb_data: HeteroData,
                               test_users: List[int],
                               candidate_items: Optional[List[List[int]]] = None,
                               k: int = 20) -> List[List[int]]:
        """
        为测试用户生成推荐
        
        Args:
            ub_data: 用户-书籍交互图数据
            bb_data: 书籍-书籍相似图数据
            test_users: 测试用户列表
            candidate_items: 候选物品列表，每个用户一个列表
            k: 推荐数量
        
        Returns:
            推荐物品列表，每个用户一个列表
        """
        self.model.eval()
        recommendations = []
        
        num_items = ub_data['book'].x.size(0)
        
        with torch.no_grad():
            for user_idx in test_users:
                if candidate_items is not None and user_idx < len(candidate_items):
                    candidates = candidate_items[user_idx]
                else:
                    candidates = list(range(num_items))
                
                if len(candidates) == 0:
                    recommendations.append([])
                    continue
                
                # 分批预测
                user_scores = []
                
                for start_idx in range(0, len(candidates), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(candidates))
                    batch_items = candidates[start_idx:end_idx]
                    
                    batch_users = [user_idx] * len(batch_items)
                    
                    user_tensor = torch.LongTensor(batch_users).to(self.device)
                    item_tensor = torch.LongTensor(batch_items).to(self.device)
                    
                    # 预测得分
                    scores = self.model(
                        ub_data.to(self.device), 
                        bb_data.to(self.device),
                        user_tensor, 
                        item_tensor
                    )
                    
                    user_scores.extend(scores.cpu().numpy())
                
                # 排序并取Top-K
                item_scores = list(zip(candidates, user_scores))
                item_scores.sort(key=lambda x: x[1], reverse=True)
                
                top_k_items = [item for item, score in item_scores[:k]]
                recommendations.append(top_k_items)
        
        return recommendations
    
    def evaluate_on_test_set(self,
                           ub_data: HeteroData,
                           bb_data: HeteroData,
                           test_data: Dict,
                           k: int = 20,
                           item_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        Args:
            ub_data: 用户-书籍交互图数据
            bb_data: 书籍-书籍相似图数据
            test_data: 测试数据
            k: 推荐数量
            item_features: 物品特征矩阵
        
        Returns:
            评估结果字典
        """
        print("在测试集上评估模型...")
        
        # 准备测试数据
        test_users = test_data['user_indices']
        test_items = test_data['item_indices']
        test_ratings = test_data['ratings']
        
        # 构建用户的相关物品列表
        user_relevant_items = defaultdict(list)
        for user, item, rating in zip(test_users, test_items, test_ratings):
            if rating > 0:  # 只考虑正反馈
                user_relevant_items[user].append(item)
        
        # 获取唯一用户
        unique_users = list(user_relevant_items.keys())
        
        # 生成推荐
        recommendations = self.generate_recommendations(
            ub_data, bb_data, unique_users, k=k
        )
        
        # 准备评估数据
        recommended_items = recommendations
        relevant_items = [user_relevant_items[user] for user in unique_users]
        
        # 计算物品流行度
        num_items = ub_data['book'].x.size(0)
        item_counts = np.zeros(num_items)
        for item in test_items:
            if item < num_items:
                item_counts[item] += 1
        
        total_interactions = len(test_items)
        item_popularity = item_counts / max(total_interactions, 1)
        
        # 评估
        results = self.evaluator.evaluate_ranking(
            recommended_items=recommended_items,
            relevant_items=relevant_items,
            total_items=num_items,
            item_features=item_features,
            item_popularity=item_popularity
        )
        
        # 评分预测评估（如果有显式评分）
        explicit_mask = test_ratings > 0
        if np.any(explicit_mask):
            explicit_users = test_users[explicit_mask]
            explicit_items = test_items[explicit_mask]
            explicit_ratings = test_ratings[explicit_mask]
            
            # 预测评分
            predictions = []
            
            with torch.no_grad():
                for start_idx in range(0, len(explicit_users), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(explicit_users))
                    
                    batch_users = explicit_users[start_idx:end_idx]
                    batch_items = explicit_items[start_idx:end_idx]
                    
                    user_tensor = torch.LongTensor(batch_users).to(self.device)
                    item_tensor = torch.LongTensor(batch_items).to(self.device)
                    
                    pred_scores = self.model(
                        ub_data.to(self.device), 
                        bb_data.to(self.device),
                        user_tensor, 
                        item_tensor
                    )
                    
                    predictions.extend(pred_scores.cpu().numpy())
            
            predictions = np.array(predictions)
            
            # 评估评分预测
            rating_results = self.evaluator.evaluate_rating_prediction(
                predictions, explicit_ratings
            )
            
            results.update(rating_results)
        
        return results
    
    def compare_models(self,
                      models: Dict[str, torch.nn.Module],
                      ub_data: HeteroData,
                      bb_data: HeteroData,
                      test_data: Dict,
                      k: int = 20) -> pd.DataFrame:
        """
        比较多个模型的性能
        
        Args:
            models: 模型字典，键为模型名称，值为模型对象
            ub_data: 用户-书籍交互图数据
            bb_data: 书籍-书籍相似图数据
            test_data: 测试数据
            k: 推荐数量
        
        Returns:
            比较结果DataFrame
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"评估模型: {model_name}")
            
            # 临时替换模型
            original_model = self.model
            self.model = model.to(self.device)
            
            # 评估
            model_results = self.evaluate_on_test_set(
                ub_data, bb_data, test_data, k
            )
            
            results[model_name] = model_results
            
            # 恢复原模型
            self.model = original_model
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results).T
        
        return results_df


if __name__ == "__main__":
    # 测试评估器
    print("测试评估器...")
    
    # 创建示例数据
    recommended_items = [
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10],
        [1, 2, 3, 4, 5]
    ]
    
    relevant_items = [
        [1, 5, 11],
        [2, 6, 12],
        [1, 3, 13]
    ]
    
    evaluator = RecommendationEvaluator()
    
    # 测试各种指标
    precision_5 = evaluator.precision_at_k(recommended_items, relevant_items, 5)
    recall_5 = evaluator.recall_at_k(recommended_items, relevant_items, 5)
    ndcg_5 = evaluator.ndcg_at_k(recommended_items, relevant_items, 5)
    map_5 = evaluator.map_at_k(recommended_items, relevant_items, 5)
    mrr = evaluator.mrr(recommended_items, relevant_items)
    
    print(f"Precision@5: {precision_5:.4f}")
    print(f"Recall@5: {recall_5:.4f}")
    print(f"NDCG@5: {ndcg_5:.4f}")
    print(f"MAP@5: {map_5:.4f}")
    print(f"MRR: {mrr:.4f}")
    
    print("评估器测试完成!")