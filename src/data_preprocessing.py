"""
数据预处理模块
包含Book-Crossing数据集的加载、清洗、特征工程等功能
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import torch

warnings.filterwarnings('ignore')

class BookCrossingDataProcessor:
    """Book-Crossing数据集预处理器"""
    
    def __init__(self, data_path: str = "data/books/"):
        self.data_path = data_path
        self.users_df = None
        self.books_df = None
        self.ratings_df = None
        
        # 映射字典
        self.user_id_map = {}
        self.book_id_map = {}
        self.reverse_user_map = {}
        self.reverse_book_map = {}
        
        # 特征处理器
        self.age_scaler = MinMaxScaler()
        self.year_scaler = MinMaxScaler()
        self.location_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.publisher_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # 下载NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def load_data(self) -> None:
        """加载原始数据"""
        print("正在加载Book-Crossing数据集...")
        
        try:
            # 加载用户数据
            self.users_df = pd.read_csv(
                f"{self.data_path}Users.csv", 
                encoding='latin-1', 
                sep=';',
                dtype={'User-ID': int, 'Age': str}
            )
            
            # 加载书籍数据
            self.books_df = pd.read_csv(
                f"{self.data_path}Books.csv", 
                encoding='latin-1', 
                sep=';',
                dtype={'Year-Of-Publication': str}
            )
            
            # 加载评分数据
            self.ratings_df = pd.read_csv(
                f"{self.data_path}Ratings.csv", 
                encoding='latin-1', 
                sep=';',
                dtype={'User-ID': int, 'Book-Rating': int}
            )
            
            print(f"用户数据: {self.users_df.shape}")
            print(f"书籍数据: {self.books_df.shape}")
            print(f"评分数据: {self.ratings_df.shape}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建示例数据用于测试
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据用于测试"""
        print("创建示例数据...")
        
        # 创建示例用户数据
        self.users_df = pd.DataFrame({
            'User-ID': range(1, 1001),
            'Location': ['New York, USA'] * 200 + ['London, UK'] * 200 + 
                       ['Tokyo, Japan'] * 200 + ['Berlin, Germany'] * 200 + 
                       ['Sydney, Australia'] * 200,
            'Age': np.random.choice([18, 25, 30, 35, 40, 45, 50, 55, 60], 1000)
        })
        
        # 创建示例书籍数据
        authors = ['J.K. Rowling', 'Stephen King', 'Agatha Christie', 'Dan Brown', 'George Orwell']
        publishers = ['Penguin', 'Random House', 'HarperCollins', 'Simon & Schuster', 'Macmillan']
        
        self.books_df = pd.DataFrame({
            'ISBN': [f'ISBN{i:06d}' for i in range(1, 2001)],
            'Book-Title': [f'Book Title {i}' for i in range(1, 2001)],
            'Book-Author': np.random.choice(authors, 2000),
            'Year-Of-Publication': np.random.choice(range(1990, 2024), 2000),
            'Publisher': np.random.choice(publishers, 2000)
        })
        
        # 创建示例评分数据
        user_ids = np.random.choice(range(1, 1001), 50000)
        book_isbns = np.random.choice([f'ISBN{i:06d}' for i in range(1, 2001)], 50000)
        ratings = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50000, 
                                 p=[0.3, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
        
        self.ratings_df = pd.DataFrame({
            'User-ID': user_ids,
            'ISBN': book_isbns,
            'Book-Rating': ratings
        })
        
        print(f"示例用户数据: {self.users_df.shape}")
        print(f"示例书籍数据: {self.books_df.shape}")
        print(f"示例评分数据: {self.ratings_df.shape}")
    
    def exploratory_data_analysis(self) -> None:
        """探索性数据分析"""
        print("\n=== 探索性数据分析 ===")
        
        # 用户数据分析
        print("\n用户数据分析:")
        print(f"总用户数: {len(self.users_df)}")
        print(f"年龄缺失值: {self.users_df['Age'].isnull().sum()}")
        print(f"位置缺失值: {self.users_df['Location'].isnull().sum()}")
        
        # 书籍数据分析
        print("\n书籍数据分析:")
        print(f"总书籍数: {len(self.books_df)}")
        print(f"作者缺失值: {self.books_df['Book-Author'].isnull().sum()}")
        print(f"出版社缺失值: {self.books_df['Publisher'].isnull().sum()}")
        
        # 评分数据分析
        print("\n评分数据分析:")
        print(f"总评分数: {len(self.ratings_df)}")
        print("评分分布:")
        print(self.ratings_df['Book-Rating'].value_counts().sort_index())
        
        # 数据稀疏性分析
        unique_users = self.ratings_df['User-ID'].nunique()
        unique_books = self.ratings_df['ISBN'].nunique()
        total_interactions = len(self.ratings_df)
        possible_interactions = unique_users * unique_books
        sparsity = 1 - (total_interactions / possible_interactions)
        
        print(f"\n数据稀疏性:")
        print(f"唯一用户数: {unique_users}")
        print(f"唯一书籍数: {unique_books}")
        print(f"实际交互数: {total_interactions}")
        print(f"可能交互数: {possible_interactions}")
        print(f"稀疏性: {sparsity:.4f}")
    
    def clean_data(self) -> None:
        """数据清洗"""
        print("\n=== 数据清洗 ===")
        
        # 清洗用户数据
        self._clean_users_data()
        
        # 清洗书籍数据
        self._clean_books_data()
        
        # 清洗评分数据
        self._clean_ratings_data()
        
        print("数据清洗完成")
    
    def _clean_users_data(self) -> None:
        """清洗用户数据"""
        print("清洗用户数据...")
        
        # 处理年龄字段
        def clean_age(age):
            if pd.isna(age) or age == '':
                return np.nan
            try:
                age_val = float(age)
                # 处理异常年龄值
                if age_val < 5 or age_val > 100:
                    return np.nan
                return age_val
            except:
                return np.nan
        
        self.users_df['Age'] = self.users_df['Age'].apply(clean_age)
        
        # 用中位数填充年龄缺失值
        median_age = self.users_df['Age'].median()
        self.users_df['Age'].fillna(median_age, inplace=True)
        
        # 处理位置字段
        self.users_df['Location'].fillna('Unknown', inplace=True)
        self.users_df['Location'] = self.users_df['Location'].str.strip().str.lower()
        
        print(f"用户数据清洗后: {self.users_df.shape}")
    
    def _clean_books_data(self) -> None:
        """清洗书籍数据"""
        print("清洗书籍数据...")
        
        # 清洗ISBN
        def clean_isbn(isbn):
            if pd.isna(isbn):
                return None
            # 移除非数字和非X字符
            cleaned = re.sub(r'[^0-9X]', '', str(isbn).upper())
            # 检查长度
            if len(cleaned) in [10, 13]:
                return cleaned
            return None
        
        self.books_df['ISBN'] = self.books_df['ISBN'].apply(clean_isbn)
        
        # 移除无效ISBN的书籍
        self.books_df = self.books_df.dropna(subset=['ISBN'])
        
        # 处理出版年份
        def clean_year(year):
            if pd.isna(year):
                return np.nan
            try:
                year_val = int(float(str(year)))
                # 检查年份合理性
                if 1800 <= year_val <= 2024:
                    return year_val
                return np.nan
            except:
                return np.nan
        
        self.books_df['Year-Of-Publication'] = self.books_df['Year-Of-Publication'].apply(clean_year)
        
        # 用中位数填充年份缺失值
        median_year = self.books_df['Year-Of-Publication'].median()
        self.books_df['Year-Of-Publication'].fillna(median_year, inplace=True)
        
        # 处理文本字段
        self.books_df['Book-Title'].fillna('Unknown Title', inplace=True)
        self.books_df['Book-Author'].fillna('Unknown Author', inplace=True)
        self.books_df['Publisher'].fillna('Unknown Publisher', inplace=True)
        
        # 清理文本字段
        text_columns = ['Book-Title', 'Book-Author', 'Publisher']
        for col in text_columns:
            self.books_df[col] = self.books_df[col].str.strip()
            # 处理HTML实体
            self.books_df[col] = self.books_df[col].str.replace('&amp;', '&')
            self.books_df[col] = self.books_df[col].str.replace('&quot;', '"')
        
        print(f"书籍数据清洗后: {self.books_df.shape}")
    
    def _clean_ratings_data(self) -> None:
        """清洗评分数据"""
        print("清洗评分数据...")
        
        # 确保评分在有效范围内
        self.ratings_df = self.ratings_df[
            (self.ratings_df['Book-Rating'] >= 0) & 
            (self.ratings_df['Book-Rating'] <= 10)
        ]
        
        # 移除无效的用户ID和ISBN
        valid_users = set(self.users_df['User-ID'])
        valid_books = set(self.books_df['ISBN'])
        
        self.ratings_df = self.ratings_df[
            self.ratings_df['User-ID'].isin(valid_users) &
            self.ratings_df['ISBN'].isin(valid_books)
        ]
        
        print(f"评分数据清洗后: {self.ratings_df.shape}")
    
    def filter_data(self, min_user_interactions: int = 5, min_book_interactions: int = 5) -> None:
        """数据过滤"""
        print(f"\n=== 数据过滤 (最小用户交互: {min_user_interactions}, 最小书籍交互: {min_book_interactions}) ===")
        
        # 迭代过滤，直到稳定
        prev_users, prev_books, prev_ratings = 0, 0, 0
        iteration = 0
        
        while True:
            iteration += 1
            print(f"过滤迭代 {iteration}...")
            
            # 过滤低活跃度用户
            user_counts = self.ratings_df['User-ID'].value_counts()
            active_users = user_counts[user_counts >= min_user_interactions].index
            self.ratings_df = self.ratings_df[self.ratings_df['User-ID'].isin(active_users)]
            
            # 过滤低热度书籍
            book_counts = self.ratings_df['ISBN'].value_counts()
            popular_books = book_counts[book_counts >= min_book_interactions].index
            self.ratings_df = self.ratings_df[self.ratings_df['ISBN'].isin(popular_books)]
            
            # 更新用户和书籍数据
            self.users_df = self.users_df[self.users_df['User-ID'].isin(self.ratings_df['User-ID'].unique())]
            self.books_df = self.books_df[self.books_df['ISBN'].isin(self.ratings_df['ISBN'].unique())]
            
            current_users = len(self.users_df)
            current_books = len(self.books_df)
            current_ratings = len(self.ratings_df)
            
            print(f"  用户: {current_users}, 书籍: {current_books}, 评分: {current_ratings}")
            
            # 检查是否收敛
            if (current_users == prev_users and 
                current_books == prev_books and 
                current_ratings == prev_ratings):
                break
            
            prev_users, prev_books, prev_ratings = current_users, current_books, current_ratings
            
            if iteration > 10:  # 防止无限循环
                break
        
        print(f"过滤完成，最终数据规模:")
        print(f"  用户: {len(self.users_df)}")
        print(f"  书籍: {len(self.books_df)}")
        print(f"  评分: {len(self.ratings_df)}")
    
    def create_id_mappings(self) -> None:
        """创建ID映射"""
        print("\n=== 创建ID映射 ===")
        
        # 用户ID映射
        unique_users = sorted(self.users_df['User-ID'].unique())
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        
        # 书籍ID映射
        unique_books = sorted(self.books_df['ISBN'].unique())
        self.book_id_map = {isbn: idx for idx, isbn in enumerate(unique_books)}
        self.reverse_book_map = {idx: isbn for isbn, idx in self.book_id_map.items()}
        
        print(f"用户映射: {len(self.user_id_map)} 个用户")
        print(f"书籍映射: {len(self.book_id_map)} 个书籍")
    
    def extract_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """特征工程"""
        print("\n=== 特征工程 ===")
        
        # 提取用户特征
        user_features = self._extract_user_features()
        
        # 提取书籍特征
        book_features = self._extract_book_features()
        
        return user_features, book_features
    
    def _extract_user_features(self) -> torch.Tensor:
        """提取用户特征"""
        print("提取用户特征...")
        
        # 按映射顺序排序用户
        users_sorted = self.users_df.set_index('User-ID').loc[
            [self.reverse_user_map[i] for i in range(len(self.user_id_map))]
        ].reset_index()
        
        features = []
        
        # 年龄特征（归一化）
        ages = users_sorted['Age'].values.reshape(-1, 1)
        ages_normalized = self.age_scaler.fit_transform(ages).flatten()
        features.append(ages_normalized)
        
        # 位置特征（编码）
        locations = users_sorted['Location'].values
        location_encoded = self.location_encoder.fit_transform(locations)
        features.append(location_encoded)
        
        # 组合特征
        user_features = np.column_stack(features)
        
        print(f"用户特征维度: {user_features.shape}")
        return torch.FloatTensor(user_features)
    
    def _extract_book_features(self) -> torch.Tensor:
        """提取书籍特征"""
        print("提取书籍特征...")
        
        # 按映射顺序排序书籍
        books_sorted = self.books_df.set_index('ISBN').loc[
            [self.reverse_book_map[i] for i in range(len(self.book_id_map))]
        ].reset_index()
        
        features = []
        
        # 出版年份特征（归一化）
        years = books_sorted['Year-Of-Publication'].values.reshape(-1, 1)
        years_normalized = self.year_scaler.fit_transform(years).flatten()
        features.append(years_normalized)
        
        # 作者特征（编码）
        authors = books_sorted['Book-Author'].values
        author_encoded = self.author_encoder.fit_transform(authors)
        features.append(author_encoded)
        
        # 出版社特征（编码）
        publishers = books_sorted['Publisher'].values
        publisher_encoded = self.publisher_encoder.fit_transform(publishers)
        features.append(publisher_encoded)
        
        # 书名TF-IDF特征
        titles = books_sorted['Book-Title'].values
        title_tfidf = self._extract_title_features(titles)
        
        # 组合特征
        basic_features = np.column_stack(features)
        book_features = np.column_stack([basic_features, title_tfidf])
        
        print(f"书籍特征维度: {book_features.shape}")
        return torch.FloatTensor(book_features)
    
    def _extract_title_features(self, titles: np.ndarray) -> np.ndarray:
        """提取书名特征"""
        print("提取书名TF-IDF特征...")
        
        # 文本预处理
        def preprocess_text(text):
            if pd.isna(text):
                return ""
            # 转小写
            text = str(text).lower()
            # 移除标点符号
            text = re.sub(r'[^\w\s]', ' ', text)
            # 分词
            tokens = word_tokenize(text)
            # 移除停用词
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            return ' '.join(tokens)
        
        processed_titles = [preprocess_text(title) for title in titles]
        
        # TF-IDF向量化
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_titles)
        
        return tfidf_matrix.toarray()
    
    def get_interaction_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取交互数据"""
        print("\n=== 获取交互数据 ===")
        
        # 映射用户和书籍ID
        user_indices = self.ratings_df['User-ID'].map(self.user_id_map).values
        book_indices = self.ratings_df['ISBN'].map(self.book_id_map).values
        ratings = self.ratings_df['Book-Rating'].values
        
        print(f"交互数据: {len(user_indices)} 条记录")
        print(f"显式评分 (1-10): {np.sum(ratings > 0)} 条")
        print(f"隐式反馈 (0): {np.sum(ratings == 0)} 条")
        
        return user_indices, book_indices, ratings
    
    def process_all(self, min_user_interactions: int = 5, min_book_interactions: int = 5) -> Dict:
        """完整的数据处理流程"""
        print("开始完整的数据处理流程...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 数据清洗
        self.clean_data()
        
        # 4. 数据过滤
        self.filter_data(min_user_interactions, min_book_interactions)
        
        # 5. 创建ID映射
        self.create_id_mappings()
        
        # 6. 特征工程
        user_features, book_features = self.extract_features()
        
        # 7. 获取交互数据
        user_indices, book_indices, ratings = self.get_interaction_data()
        
        # 返回处理后的数据
        processed_data = {
            'user_features': user_features,
            'book_features': book_features,
            'user_indices': user_indices,
            'book_indices': book_indices,
            'ratings': ratings,
            'num_users': len(self.user_id_map),
            'num_books': len(self.book_id_map),
            'user_id_map': self.user_id_map,
            'book_id_map': self.book_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_book_map': self.reverse_book_map
        }
        
        print("\n数据处理完成!")
        return processed_data


def preprocess_book_crossing_data(data_path: str = "data/books/", 
                                min_user_interactions: int = 5,
                                min_book_interactions: int = 5) -> Dict:
    """
    Book-Crossing数据预处理的主函数
    
    Args:
        data_path: 数据文件路径
        min_user_interactions: 最小用户交互次数
        min_book_interactions: 最小书籍交互次数
    
    Returns:
        处理后的数据字典
    """
    processor = BookCrossingDataProcessor(data_path)
    return processor.process_all(min_user_interactions, min_book_interactions)


if __name__ == "__main__":
    # 测试数据预处理
    processed_data = preprocess_book_crossing_data()
    
    print(f"\n最终数据统计:")
    print(f"用户数: {processed_data['num_users']}")
    print(f"书籍数: {processed_data['num_books']}")
    print(f"交互数: {len(processed_data['ratings'])}")
    print(f"用户特征维度: {processed_data['user_features'].shape}")
    print(f"书籍特征维度: {processed_data['book_features'].shape}")