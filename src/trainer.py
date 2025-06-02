"""
训练模块
包含损失函数、优化器、训练循环等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
from tqdm import tqdm
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


class BPRLoss(nn.Module):
    """贝叶斯个性化排序损失"""

    def __init__(self, reg_weight: float = 1e-5):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        user_embed: Optional[torch.Tensor] = None,
        pos_item_embed: Optional[torch.Tensor] = None,
        neg_item_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算BPR损失

        Args:
            pos_scores: 正样本得分
            neg_scores: 负样本得分
            user_embed: 用户嵌入（用于正则化）
            pos_item_embed: 正样本物品嵌入（用于正则化）
            neg_item_embed: 负样本物品嵌入（用于正则化）

        Returns:
            BPR损失
        """
        # BPR损失
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        # L2正则化
        reg_loss = 0
        if user_embed is not None:
            reg_loss += torch.norm(user_embed, p=2).pow(2)
        if pos_item_embed is not None:
            reg_loss += torch.norm(pos_item_embed, p=2).pow(2)
        if neg_item_embed is not None:
            reg_loss += torch.norm(neg_item_embed, p=2).pow(2)

        reg_loss = self.reg_weight * reg_loss / pos_scores.size(0)

        return bpr_loss + reg_loss


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(
        self,
        explicit_weight: float = 0.5,
        implicit_weight: float = 0.5,
        reg_weight: float = 1e-5,
    ):
        super().__init__()
        self.explicit_weight = explicit_weight
        self.implicit_weight = implicit_weight
        self.reg_weight = reg_weight

        self.mse_loss = nn.MSELoss()
        self.bpr_loss = BPRLoss(reg_weight=0)  # BPR内部不加正则化

    def forward(
        self,
        explicit_pred: Optional[torch.Tensor] = None,
        explicit_target: Optional[torch.Tensor] = None,
        implicit_pos_scores: Optional[torch.Tensor] = None,
        implicit_neg_scores: Optional[torch.Tensor] = None,
        embeddings: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        计算多任务损失

        Args:
            explicit_pred: 显式评分预测
            explicit_target: 显式评分目标
            implicit_pos_scores: 隐式正样本得分
            implicit_neg_scores: 隐式负样本得分
            embeddings: 嵌入列表（用于正则化）

        Returns:
            总损失
        """
        total_loss = 0

        # 显式评分损失
        if explicit_pred is not None and explicit_target is not None:
            explicit_loss = self.mse_loss(explicit_pred, explicit_target)
            total_loss += self.explicit_weight * explicit_loss

        # 隐式反馈损失
        if implicit_pos_scores is not None and implicit_neg_scores is not None:
            implicit_loss = self.bpr_loss(implicit_pos_scores, implicit_neg_scores)
            total_loss += self.implicit_weight * implicit_loss

        # L2正则化
        if embeddings is not None:
            reg_loss = 0
            for embed in embeddings:
                reg_loss += torch.norm(embed, p=2).pow(2)
            reg_loss = self.reg_weight * reg_loss / len(embeddings)
            total_loss += reg_loss

        return total_loss


class NegativeSampler:
    """负采样器"""

    def __init__(
        self, num_items: int, num_negatives: int = 1, sampling_strategy: str = "random"
    ):
        """
        初始化负采样器

        Args:
            num_items: 物品总数
            num_negatives: 每个正样本对应的负样本数
            sampling_strategy: 采样策略 ('random', 'popularity')
        """
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.sampling_strategy = sampling_strategy
        self.item_popularity = None

    def set_item_popularity(self, item_counts: np.ndarray):
        """设置物品流行度（用于基于流行度的采样）"""
        self.item_popularity = item_counts / item_counts.sum()

    def sample(
        self,
        user_indices: np.ndarray,
        pos_item_indices: np.ndarray,
        user_item_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        负采样

        Args:
            user_indices: 用户索引
            pos_item_indices: 正样本物品索引
            user_item_matrix: 用户-物品交互矩阵（用于避免采样已交互物品）

        Returns:
            (扩展的用户索引, 负样本物品索引)
        """
        neg_users = []
        neg_items = []

        for i, (user_idx, pos_item_idx) in enumerate(
            zip(user_indices, pos_item_indices)
        ):
            # 为每个正样本生成负样本
            for _ in range(self.num_negatives):
                # 采样负样本物品
                if self.sampling_strategy == "random":
                    neg_item = random.randint(0, self.num_items - 1)
                elif self.sampling_strategy == "popularity":
                    if self.item_popularity is not None:
                        neg_item = np.random.choice(
                            self.num_items, p=self.item_popularity
                        )
                    else:
                        neg_item = random.randint(0, self.num_items - 1)
                else:
                    neg_item = random.randint(0, self.num_items - 1)

                # 确保负样本不是用户已交互的物品
                if user_item_matrix is not None:
                    max_attempts = 10
                    attempts = 0
                    while (
                        user_item_matrix[user_idx, neg_item] > 0
                        and attempts < max_attempts
                    ):
                        if self.sampling_strategy == "random":
                            neg_item = random.randint(0, self.num_items - 1)
                        elif self.sampling_strategy == "popularity":
                            if self.item_popularity is not None:
                                neg_item = np.random.choice(
                                    self.num_items, p=self.item_popularity
                                )
                            else:
                                neg_item = random.randint(0, self.num_items - 1)
                        attempts += 1

                neg_users.append(user_idx)
                neg_items.append(neg_item)

        return np.array(neg_users), np.array(neg_items)


class DataSplitter:
    """数据划分器"""

    def __init__(
        self,
        split_method: str = "random",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ):
        """
        初始化数据划分器

        Args:
            split_method: 划分方法 ('random', 'temporal', 'user_based')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
        """
        self.split_method = split_method
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    def split(
        self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray
    ) -> Tuple[Dict, Dict, Dict]:
        """
        划分数据

        Args:
            user_indices: 用户索引
            item_indices: 物品索引
            ratings: 评分

        Returns:
            (训练集, 验证集, 测试集)
        """
        if self.split_method == "random":
            return self._random_split(user_indices, item_indices, ratings)
        elif self.split_method == "user_based":
            return self._user_based_split(user_indices, item_indices, ratings)
        else:
            raise ValueError(f"不支持的划分方法: {self.split_method}")

    def _random_split(
        self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray
    ) -> Tuple[Dict, Dict, Dict]:
        """随机划分"""
        # 首先划分训练集和临时集
        (
            train_users,
            temp_users,
            train_items,
            temp_items,
            train_ratings,
            temp_ratings,
        ) = train_test_split(
            user_indices,
            item_indices,
            ratings,
            test_size=(1 - self.train_ratio),
            random_state=self.random_state,
            stratify=None,
        )

        # 再划分验证集和测试集
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_users, test_users, val_items, test_items, val_ratings, test_ratings = (
            train_test_split(
                temp_users,
                temp_items,
                temp_ratings,
                test_size=(1 - val_ratio_adjusted),
                random_state=self.random_state,
                stratify=None,
            )
        )

        train_data = {
            "user_indices": train_users,
            "item_indices": train_items,
            "ratings": train_ratings,
        }

        val_data = {
            "user_indices": val_users,
            "item_indices": val_items,
            "ratings": val_ratings,
        }

        test_data = {
            "user_indices": test_users,
            "item_indices": test_items,
            "ratings": test_ratings,
        }

        return train_data, val_data, test_data

    def _user_based_split(
        self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray
    ) -> Tuple[Dict, Dict, Dict]:
        """基于用户的划分（每个用户的交互按比例划分）"""
        train_users, train_items, train_ratings = [], [], []
        val_users, val_items, val_ratings = [], [], []
        test_users, test_items, test_ratings = [], [], []

        # 按用户分组
        unique_users = np.unique(user_indices)

        for user in unique_users:
            user_mask = user_indices == user
            user_items = item_indices[user_mask]
            user_ratings = ratings[user_mask]

            n_interactions = len(user_items)

            if n_interactions < 3:
                # 如果交互太少，全部放入训练集
                train_users.extend([user] * n_interactions)
                train_items.extend(user_items)
                train_ratings.extend(user_ratings)
            else:
                # 按比例划分
                n_train = max(1, int(n_interactions * self.train_ratio))
                n_val = max(1, int(n_interactions * self.val_ratio))
                n_test = n_interactions - n_train - n_val

                # 随机打乱
                indices = np.random.RandomState(self.random_state).permutation(
                    n_interactions
                )

                # 划分
                train_idx = indices[:n_train]
                val_idx = indices[n_train : n_train + n_val]
                test_idx = indices[n_train + n_val :]

                # 添加到对应集合
                train_users.extend([user] * len(train_idx))
                train_items.extend(user_items[train_idx])
                train_ratings.extend(user_ratings[train_idx])

                if len(val_idx) > 0:
                    val_users.extend([user] * len(val_idx))
                    val_items.extend(user_items[val_idx])
                    val_ratings.extend(user_ratings[val_idx])

                if len(test_idx) > 0:
                    test_users.extend([user] * len(test_idx))
                    test_items.extend(user_items[test_idx])
                    test_ratings.extend(user_ratings[test_idx])

        train_data = {
            "user_indices": np.array(train_users),
            "item_indices": np.array(train_items),
            "ratings": np.array(train_ratings),
        }

        val_data = {
            "user_indices": np.array(val_users),
            "item_indices": np.array(val_items),
            "ratings": np.array(val_ratings),
        }

        test_data = {
            "user_indices": np.array(test_users),
            "item_indices": np.array(test_items),
            "ratings": np.array(test_ratings),
        }

        return train_data, val_data, test_data


class Trainer:
    """训练器"""

    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        num_negatives: int = 1,
        loss_type: str = "bpr",
        explicit_weight: float = 0.5,
        implicit_weight: float = 0.5,
    ):
        """
        初始化训练器

        Args:
            model: 模型
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            batch_size: 批大小
            num_negatives: 负样本数
            loss_type: 损失类型 ('bpr', 'mse', 'multitask')
            explicit_weight: 显式反馈权重
            implicit_weight: 隐式反馈权重
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.loss_type = loss_type

        # 优化器
        self.optimizer = Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # 损失函数
        if loss_type == "bpr":
            self.criterion = BPRLoss(reg_weight=0)  # 正则化在优化器中处理
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "multitask":
            self.criterion = MultiTaskLoss(
                explicit_weight=explicit_weight,
                implicit_weight=implicit_weight,
                reg_weight=0,
            )
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

        # 负采样器
        self.negative_sampler = None

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def set_negative_sampler(self, num_items: int, sampling_strategy: str = "random"):
        """设置负采样器"""
        self.negative_sampler = NegativeSampler(
            num_items=num_items,
            num_negatives=self.num_negatives,
            sampling_strategy=sampling_strategy,
        )

    def train_epoch(
        self, ub_data: HeteroData, bb_data: HeteroData, train_data: Dict
    ) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # 准备训练数据
        user_indices = train_data["user_indices"]
        item_indices = train_data["item_indices"]
        ratings = train_data["ratings"]

        # 创建用户-物品交互矩阵（用于负采样）
        num_users = ub_data["user"].x.size(0)
        num_items = ub_data["book"].x.size(0)
        user_item_matrix = np.zeros((num_users, num_items))
        user_item_matrix[user_indices, item_indices] = 1

        # 分批训练
        n_samples = len(user_indices)
        indices = np.random.permutation(n_samples)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_users = user_indices[batch_indices]
            batch_items = item_indices[batch_indices]
            batch_ratings = ratings[batch_indices]

            # 移动到设备
            batch_users_tensor = torch.LongTensor(batch_users).to(self.device)
            batch_items_tensor = torch.LongTensor(batch_items).to(self.device)
            batch_ratings_tensor = torch.FloatTensor(batch_ratings).to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            if self.loss_type == "mse":
                # MSE损失：直接预测评分
                predictions = self.model(
                    ub_data.to(self.device),
                    bb_data.to(self.device),
                    batch_users_tensor,
                    batch_items_tensor,
                )
                loss = self.criterion(predictions, batch_ratings_tensor)

            elif self.loss_type == "bpr":
                # BPR损失：需要负采样
                if self.negative_sampler is None:
                    self.set_negative_sampler(num_items)

                # 负采样
                neg_users, neg_items = self.negative_sampler.sample(
                    batch_users, batch_items, user_item_matrix
                )

                neg_users_tensor = torch.LongTensor(neg_users).to(self.device)
                neg_items_tensor = torch.LongTensor(neg_items).to(self.device)

                # 计算正负样本得分
                pos_scores = self.model(
                    ub_data.to(self.device),
                    bb_data.to(self.device),
                    batch_users_tensor,
                    batch_items_tensor,
                )

                neg_scores = self.model(
                    ub_data.to(self.device),
                    bb_data.to(self.device),
                    neg_users_tensor,
                    neg_items_tensor,
                )

                loss = self.criterion(pos_scores, neg_scores)

            elif self.loss_type == "multitask":
                # 多任务损失：区分显式和隐式反馈
                explicit_mask = batch_ratings > 0
                implicit_mask = batch_ratings == 0

                explicit_loss_val = None
                implicit_loss_val = None

                # 显式评分损失
                if explicit_mask.any():
                    explicit_users = batch_users_tensor[explicit_mask]
                    explicit_items = batch_items_tensor[explicit_mask]
                    explicit_ratings = batch_ratings_tensor[explicit_mask]

                    explicit_pred = self.model(
                        ub_data.to(self.device),
                        bb_data.to(self.device),
                        explicit_users,
                        explicit_items,
                    )
                    explicit_loss_val = (explicit_pred, explicit_ratings)

                # 隐式反馈损失
                if implicit_mask.any():
                    implicit_users = batch_users[implicit_mask]
                    implicit_items = batch_items[implicit_mask]

                    if len(implicit_users) > 0 and self.negative_sampler is not None:
                        # 负采样
                        neg_users, neg_items = self.negative_sampler.sample(
                            implicit_users, implicit_items, user_item_matrix
                        )

                        implicit_users_tensor = torch.LongTensor(implicit_users).to(
                            self.device
                        )
                        implicit_items_tensor = torch.LongTensor(implicit_items).to(
                            self.device
                        )
                        neg_users_tensor = torch.LongTensor(neg_users).to(self.device)
                        neg_items_tensor = torch.LongTensor(neg_items).to(self.device)

                        # 计算正负样本得分
                        pos_scores = self.model(
                            ub_data.to(self.device),
                            bb_data.to(self.device),
                            implicit_users_tensor,
                            implicit_items_tensor,
                        )

                        neg_scores = self.model(
                            ub_data.to(self.device),
                            bb_data.to(self.device),
                            neg_users_tensor,
                            neg_items_tensor,
                        )

                        implicit_loss_val = (pos_scores, neg_scores)

                # 计算多任务损失
                if explicit_loss_val is not None and implicit_loss_val is not None:
                    loss = self.criterion(
                        explicit_pred=explicit_loss_val[0],
                        explicit_target=explicit_loss_val[1],
                        implicit_pos_scores=implicit_loss_val[0],
                        implicit_neg_scores=implicit_loss_val[1],
                    )
                elif explicit_loss_val is not None:
                    loss = self.criterion(
                        explicit_pred=explicit_loss_val[0],
                        explicit_target=explicit_loss_val[1],
                    )
                elif implicit_loss_val is not None:
                    loss = self.criterion(
                        implicit_pos_scores=implicit_loss_val[0],
                        implicit_neg_scores=implicit_loss_val[1],
                    )
                else:
                    continue  # 跳过这个批次

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(
        self, ub_data: HeteroData, bb_data: HeteroData, eval_data: Dict
    ) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        user_indices = eval_data["user_indices"]
        item_indices = eval_data["item_indices"]
        ratings = eval_data["ratings"]

        with torch.no_grad():
            n_samples = len(user_indices)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)

                batch_users = user_indices[start_idx:end_idx]
                batch_items = item_indices[start_idx:end_idx]
                batch_ratings = ratings[start_idx:end_idx]

                batch_users_tensor = torch.LongTensor(batch_users).to(self.device)
                batch_items_tensor = torch.LongTensor(batch_items).to(self.device)
                batch_ratings_tensor = torch.FloatTensor(batch_ratings).to(self.device)

                # 预测
                predictions = self.model(
                    ub_data.to(self.device),
                    bb_data.to(self.device),
                    batch_users_tensor,
                    batch_items_tensor,
                )

                # 计算损失（使用MSE作为评估指标）
                loss = F.mse_loss(predictions, batch_ratings_tensor)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        ub_data: HeteroData,
        bb_data: HeteroData,
        train_data: Dict,
        val_data: Dict,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """训练模型"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练样本数: {len(train_data['user_indices'])}")
        print(f"验证样本数: {len(val_data['user_indices'])}")

        # 设置负采样器
        if self.loss_type in ["bpr", "multitask"]:
            num_items = ub_data["book"].x.size(0)
            self.set_negative_sampler(num_items)

        best_epoch = 0
        patience_counter = 0

        for epoch in tqdm(range(num_epochs)):
            start_time = time.time()

            # 训练
            train_loss = self.train_epoch(ub_data, bb_data, train_data)

            # 验证
            val_loss = self.evaluate(ub_data, bb_data, val_data)

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 学习率调度
            self.scheduler.step(val_loss)

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            epoch_time = time.time() - start_time

            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # 早停
            if patience_counter >= early_stopping_patience:
                print(
                    f"早停于第 {epoch+1} 轮，最佳验证损失: {self.best_val_loss:.4f} (第 {best_epoch+1} 轮)"
                )
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        training_history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": best_epoch,
        }

        return training_history


if __name__ == "__main__":
    # 测试训练器
    print("测试训练器...")

    # 这里需要实际的数据和模型来测试
    # 由于依赖较多，这里只是展示接口
    print("训练器模块定义完成!")
