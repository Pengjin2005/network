# 基于GNN和Book-Crossing数据集的双重网络书籍推荐算法

本项目实现了一个基于图神经网络(GNN)的双重网络书籍推荐系统，使用Book-Crossing数据集进行训练和评估。该系统通过融合用户-书籍交互图和书籍-书籍相似图的信息，提供更准确和鲁棒的推荐结果。

## 🌟 主要特性

- **双重网络架构**: 同时利用用户-书籍交互图和书籍-书籍相似图
- **多种GNN模型支持**: LightGCN、GraphSAGE、GAT、GCN
- **灵活的融合策略**: 注意力机制、门控机制、拼接、平均等
- **多任务学习**: 同时处理显式评分和隐式反馈
- **全面的评估指标**: Precision@K、Recall@K、NDCG@K、MRR、覆盖率、多样性等
- **完整的数据处理流程**: 从原始数据到模型训练的端到端实现

## 📋 系统要求

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- 其他依赖见 `pyproject.toml`

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用uv (推荐)
uv sync

# 或使用pip
pip install -e .
```

### 2. 准备数据

将Book-Crossing数据集文件放置在 `data/books/` 目录下：
- `Users.csv`: 用户信息
- `Books.csv`: 书籍信息  
- `Ratings.csv`: 评分数据

如果没有真实数据，系统会自动生成示例数据用于测试。

### 3. 运行系统

```bash
# 使用默认配置
python main.py

# 使用自定义配置
python main.py --config config.json --save_dir results

# 指定数据路径
python main.py --data_path data/books/ --save_dir my_results
```

## 📁 项目结构

```
network/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── data_preprocessing.py     # 数据预处理模块
│   ├── graph_builder.py          # 图构建模块
│   ├── dual_network_gnn.py       # 双重网络GNN模型
│   ├── trainer.py                # 训练模块
│   ├── evaluator.py              # 评估模块
│   └── main.py                   # 主程序
├── data/                         # 数据目录
│   └── books/                    # Book-Crossing数据集
├── main.py                       # 程序入口
├── config.json                   # 配置文件
├── pyproject.toml               # 项目配置
└── README.md                    # 说明文档
```

## ⚙️ 配置说明

主要配置参数（`config.json`）：

### 数据相关
- `data_path`: 数据文件路径
- `min_user_interactions`: 最小用户交互次数
- `min_book_interactions`: 最小书籍交互次数

### 图构建相关
- `ub_feedback_handling`: 用户-书籍图反馈处理方式
  - `"separate"`: 区分显式和隐式反馈
  - `"unified"`: 统一为隐式交互
  - `"explicit_only"`: 仅使用显式反馈
- `bb_similarity_method`: 书籍相似性计算方法
  - `"content"`: 基于内容特征
  - `"collaborative"`: 基于协同信号
  - `"hybrid"`: 混合方法

### 模型相关
- `embed_dim`: 嵌入维度
- `ub_gnn_type`: 用户-书籍GNN类型 (`"lightgcn"`, `"sage"`, `"gat"`, `"gcn"`)
- `bb_gnn_type`: 书籍-书籍GNN类型 (`"sage"`, `"gat"`, `"gcn"`)
- `fusion_type`: 融合策略 (`"attention"`, `"gate"`, `"concat"`, `"average"`)

### 训练相关
- `learning_rate`: 学习率
- `batch_size`: 批大小
- `num_epochs`: 训练轮次
- `loss_type`: 损失函数类型 (`"bpr"`, `"mse"`, `"multitask"`)

## 🔬 算法原理

### 双重网络架构

1. **用户-书籍交互图 (U-B Graph)**
   - 节点: 用户和书籍
   - 边: 用户对书籍的评分或交互
   - 目标: 学习协同过滤信号

2. **书籍-书籍相似图 (B-B Graph)**
   - 节点: 书籍
   - 边: 基于内容或协同信号的相似性
   - 目标: 学习书籍的内容表示

### 模型流程

1. **数据预处理**: 清洗、过滤、特征工程
2. **图构建**: 构建双重异构图
3. **GNN编码**: 分别使用不同GNN编码两个图
4. **嵌入融合**: 融合来自两个图的书籍嵌入
5. **预测**: 计算用户对书籍的偏好得分

### 关键创新点

- **多源信息融合**: 结合协同过滤和内容信息
- **异构图处理**: 支持多种节点和边类型
- **自适应融合**: 使用注意力机制动态调整权重
- **多任务学习**: 同时优化评分预测和排序任务

## 📊 评估指标

### 排序指标
- **Precision@K**: Top-K推荐的精确率
- **Recall@K**: Top-K推荐的召回率
- **NDCG@K**: 归一化折损累积增益
- **MAP@K**: 平均精确率均值
- **MRR**: 平均倒数排名

### 其他指标
- **Coverage**: 推荐覆盖率
- **Diversity**: 推荐多样性
- **Novelty**: 推荐新颖性
- **RMSE/MAE**: 评分预测误差

## 🎯 使用示例

### 基本使用

```python
from src.main import DualNetworkRecommendationSystem, get_default_config

# 创建配置
config = get_default_config()
config['embed_dim'] = 128
config['num_epochs'] = 50

# 创建推荐系统
system = DualNetworkRecommendationSystem(config)

# 运行完整流程
results = system.run_full_pipeline(save_dir='my_results')

print(f"NDCG@10: {results['NDCG@10']:.4f}")
```

### 自定义模型

```python
from src.dual_network_gnn import DualNetworkGNN

# 创建自定义模型
model = DualNetworkGNN(
    user_feature_dim=10,
    book_feature_dim=50,
    embed_dim=64,
    ub_gnn_type='lightgcn',
    bb_gnn_type='gat',
    fusion_type='attention'
)
```

## 📈 实验结果

系统会自动生成包含以下内容的结果报告：

1. **训练过程可视化**: 损失曲线、收敛情况
2. **性能指标对比**: 各种评估指标的数值和图表
3. **模型分析**: 参数统计、计算复杂度
4. **结果保存**: 模型权重、配置文件、评估结果

## 🛠️ 扩展功能

### 添加新的GNN模型

```python
# 在dual_network_gnn.py中添加新的GNN层
class CustomGNN(nn.Module):
    def __init__(self, ...):
        # 实现自定义GNN
        pass
    
    def forward(self, ...):
        # 前向传播逻辑
        pass
```

### 自定义融合策略

```python
# 在EmbeddingFusion类中添加新的融合方法
def custom_fusion(self, collab_embed, content_embed):
    # 实现自定义融合逻辑
    return fused_embed
```

### 新的评估指标

```python
# 在evaluator.py中添加新的评估函数
def custom_metric(self, recommended_items, relevant_items):
    # 计算自定义指标
    return metric_value
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 降低 `embed_dim`
   - 使用更小的数据子集

2. **训练不收敛**
   - 调整 `learning_rate`
   - 增加 `early_stopping_patience`
   - 检查数据质量

3. **GPU相关问题**
   - 确保PyTorch和CUDA版本兼容
   - 检查GPU内存使用情况

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用小数据集测试
config['min_user_interactions'] = 2
config['min_book_interactions'] = 2
config['num_epochs'] = 5
```

## 📚 参考文献

1. He, X., et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
2. Hamilton, W., et al. "Inductive Representation Learning on Large Graphs." NIPS 2017.
3. Veličković, P., et al. "Graph Attention Networks." ICLR 2018.
4. Wang, X., et al. "Neural Graph Collaborative Filtering." SIGIR 2019.

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本项目仅用于学术研究和教育目的。在生产环境中使用前，请进行充分的测试和验证。