"""
负样本难度分析器模块
用于分析视频时刻检索任务中负查询与正查询的语义相似度，
并将负样本分类为简易负样本(Easy)和困难负样本(Hard)。
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class NegativeSample:
    """负样本数据结构"""
    sentence: str
    video_id: str  # 负查询所属的视频ID
    source_sentence: str  # 正查询内容
    source_video_id: str  # 正查询所属的视频ID
    similarity_score: float  # 相似度分数 0-1
    difficulty: str  # 'easy' or 'hard'
    id: str  # 样本ID


class LLMClient:
    """LLM客户端基类，支持多种LLM API"""

    def __init__(self, api_key: str, base_url: str, model: str = "qwen-plus"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None

    def _get_client(self):
        """延迟初始化客户端"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def get_similarity(self, query1: str, query2: str) -> float:
        """
        使用LLM计算两个查询的语义相似度

        Args:
            query1: 第一个查询文本
            query2: 第二个查询文本

        Returns:
            相似度分数，范围0-1
        """
        client = self._get_client()

        prompt = f"""Please analyze the semantic similarity between the following two video moment retrieval queries.

Query 1: "{query1}"
Query 2: "{query2}"

Analyze the similarity in terms of:
1. Semantic meaning (actions, objects, scenes)
2. Intent similarity
3. Potential confusion likelihood

Provide a similarity score between 0 and 1, where:
- 0.0 - 0.3: Completely different (different actions, objects, scenes)
- 0.3 - 0.5: Somewhat related but clearly distinguishable
- 0.5 - 0.7: Related with moderate similarity
- 0.7 - 0.9: Highly similar, potentially confusing
- 0.9 - 1.0: Nearly identical, very hard to distinguish

Respond with ONLY a single number (e.g., 0.75), no explanation."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a semantic similarity analyzer for video retrieval queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # 低温度保证结果稳定
                max_tokens=10
            )

            result = response.choices[0].message.content.strip()
            # 提取数字
            score = float(result)
            return max(0.0, min(1.0, score))  # 确保在0-1范围内

        except Exception as e:
            print(f"Error getting similarity: {e}")
            return 0.0


class EmbeddingClient:
    """基于嵌入的相似度计算客户端（更快速、成本更低）"""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v3"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None
        self._cache = {}  # 缓存嵌入向量

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        if text in self._cache:
            return self._cache[text]

        client = self._get_client()

        try:
            response = client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            self._cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 1536  # 返回零向量

    def get_similarity(self, query1: str, query2: str) -> float:
        """
        使用余弦相似度计算两个查询的相似度

        Args:
            query1: 第一个查询文本
            query2: 第二个查询文本

        Returns:
            相似度分数，范围0-1
        """
        emb1 = self._get_embedding(query1)
        emb2 = self._get_embedding(query2)

        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)


class NegativeAnalyzer:
    """负样本难度分析器"""

    # 相似度阈值设置
    DEFAULT_HARD_THRESHOLD = 0.6  # 高于此阈值视为困难负样本
    DEFAULT_EASY_THRESHOLD = 0.4  # 低于此阈值视为简易负样本

    def __init__(
        self,
        similarity_client,
        hard_threshold: float = DEFAULT_HARD_THRESHOLD,
        easy_threshold: float = DEFAULT_EASY_THRESHOLD,
        use_cache: bool = True
    ):
        """
        初始化负样本分析器

        Args:
            similarity_client: 相似度计算客户端（LLMClient或EmbeddingClient）
            hard_threshold: 困难负样本的相似度阈值
            easy_threshold: 简易负样本的相似度阈值
            use_cache: 是否使用缓存
        """
        self.similarity_client = similarity_client
        self.hard_threshold = hard_threshold
        self.easy_threshold = easy_threshold
        self.use_cache = use_cache
        self._similarity_cache = {}

    def _get_similarity(self, query1: str, query2: str) -> float:
        """获取相似度，带缓存"""
        cache_key = f"{query1}|||{query2}"
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        similarity = self.similarity_client.get_similarity(query1, query2)

        if self.use_cache:
            self._similarity_cache[cache_key] = similarity

        return similarity

    def analyze_dataset(
        self,
        data: Dict,
        batch_size: int = 10,
        save_intermediate: bool = True,
        output_dir: str = None
    ) -> Dict[str, List[NegativeSample]]:
        """
        分析整个数据集

        Args:
            data: 原始数据集（从JSON加载）
            batch_size: 批处理大小
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录

        Returns:
            分类结果: {
                'easy': [NegativeSample],
                'hard': [NegativeSample],
                'medium': [NegativeSample]
            }
        """
        results = {
            'easy': [],
            'hard': [],
            'medium': []
        }

        # 收集所有正查询（按视频分组）
        positive_queries = defaultdict(list)
        for video_id, video_data in data.items():
            for item in video_data.get("sts", []):
                if not item.get("no_answer", False):
                    positive_queries[video_id].append({
                        'sentence': item['sentence'],
                        'id': item['id']
                    })

        # 分析每个负查询
        total_negatives = 0
        processed = 0

        for video_id, video_data in data.items():
            for item in video_data.get("sts", []):
                if item.get("no_answer", False):
                    total_negatives += 1

        print(f"Found {total_negatives} negative samples to analyze...")

        for video_id, video_data in data.items():
            # 获取当前视频的所有正查询
            current_video_positives = positive_queries.get(video_id, [])

            # 如果当前视频没有正查询，跳过
            if not current_video_positives:
                continue

            for item in video_data.get("sts", []):
                if item.get("no_answer", False):
                    negative_sentence = item['sentence']
                    negative_id = item['id']

                    # 计算与当前视频所有正查询的最大相似度
                    max_similarity = 0.0
                    most_similar_positive = ""

                    for pos in current_video_positives:
                        similarity = self._get_similarity(
                            negative_sentence,
                            pos['sentence']
                        )
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_positive = pos['sentence']

                    # 根据相似度分类
                    if max_similarity >= self.hard_threshold:
                        difficulty = 'hard'
                    elif max_similarity <= self.easy_threshold:
                        difficulty = 'easy'
                    else:
                        difficulty = 'medium'

                    neg_sample = NegativeSample(
                        sentence=negative_sentence,
                        video_id=video_id,
                        source_sentence=most_similar_positive,
                        source_video_id=video_id,
                        similarity_score=max_similarity,
                        difficulty=difficulty,
                        id=negative_id
                    )

                    results[difficulty].append(neg_sample)

                    processed += 1
                    if processed % batch_size == 0:
                        print(f"Processed {processed}/{total_negatives} samples...")

                    # 保存中间结果
                    if save_intermediate and processed % 100 == 0 and output_dir:
                        self._save_intermediate_results(
                            results, processed, output_dir
                        )

        print(f"\nAnalysis complete!")
        print(f"Easy negatives: {len(results['easy'])}")
        print(f"Medium negatives: {len(results['medium'])}")
        print(f"Hard negatives: {len(results['hard'])}")

        return results

    def _save_intermediate_results(
        self,
        results: Dict,
        processed: int,
        output_dir: str
    ):
        """保存中间结果"""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"intermediate_results_{processed}.json"
        )

        serializable_results = {
            key: [self._negative_sample_to_dict(ns) for ns in samples]
            for key, samples in results.items()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    def _negative_sample_to_dict(self, ns: NegativeSample) -> Dict:
        """将NegativeSample转换为字典"""
        return {
            'sentence': ns.sentence,
            'video_id': ns.video_id,
            'source_sentence': ns.source_sentence,
            'source_video_id': ns.source_video_id,
            'similarity_score': ns.similarity_score,
            'difficulty': ns.difficulty,
            'id': ns.id
        }

    def save_results(self, results: Dict, output_file: str):
        """保存最终结果"""
        serializable_results = {
            key: [self._negative_sample_to_dict(ns) for ns in samples]
            for key, samples in results.items()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {output_file}")

    def load_results(self, input_file: str) -> Dict[str, List[NegativeSample]]:
        """加载已保存的结果"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = {}
        for key, samples in data.items():
            results[key] = [
                NegativeSample(
                    sentence=s['sentence'],
                    video_id=s['video_id'],
                    source_sentence=s['source_sentence'],
                    source_video_id=s['source_video_id'],
                    similarity_score=s['similarity_score'],
                    difficulty=s['difficulty'],
                    id=s['id']
                )
                for s in samples
            ]

        return results


class NegativeDatasetSplitter:
    """负样本数据集分割器"""

    @staticmethod
    def split_by_difficulty(
        original_data: Dict,
        analysis_results: Dict[str, List[NegativeSample]],
        include_medium: str = 'hard'  # 'hard', 'easy', or 'both'
    ) -> Tuple[Dict, Dict]:
        """
        根据难度将原始数据集分割为简易和困难两个版本

        Args:
            original_data: 原始数据集
            analysis_results: 分析结果
            include_medium: 如何处理medium样本

        Returns:
            (easy_dataset, hard_dataset)
        """
        easy_data = {}
        hard_data = {}

        # 收集简易和困难负样本的ID
        easy_ids = set()
        hard_ids = set()

        for ns in analysis_results.get('easy', []):
            easy_ids.add(ns.id)

        for ns in analysis_results.get('hard', []):
            hard_ids.add(ns.id)

        # 处理medium
        medium_ids = set()
        for ns in analysis_results.get('medium', []):
            medium_ids.add(ns.id)

        if include_medium == 'hard':
            hard_ids.update(medium_ids)
        elif include_medium == 'easy':
            easy_ids.update(medium_ids)
        elif include_medium == 'both':
            # 分配到两边
            pass

        # 构建简易数据集
        for video_id, video_data in original_data.items():
            easy_sts = []
            for item in video_data.get("sts", []):
                # 保留所有正样本
                if not item.get("no_answer", False):
                    easy_sts.append(item)
                # 只保留简易负样本
                elif item['id'] in easy_ids:
                    easy_sts.append(item)

            if easy_sts:
                easy_data[video_id] = {
                    "sts": easy_sts,
                    "duration": video_data["duration"]
                }

        # 构建困难数据集
        for video_id, video_data in original_data.items():
            hard_sts = []
            for item in video_data.get("sts", []):
                # 保留所有正样本
                if not item.get("no_answer", False):
                    hard_sts.append(item)
                # 只保留困难负样本
                elif item['id'] in hard_ids:
                    hard_sts.append(item)

            if hard_sts:
                hard_data[video_id] = {
                    "sts": hard_sts,
                    "duration": video_data["duration"]
                }

        return easy_data, hard_data

    @staticmethod
    def create_annotated_dataset(
        original_data: Dict,
        analysis_results: Dict[str, List[NegativeSample]]
    ) -> Dict:
        """
        创建带有难度标注的数据集

        Args:
            original_data: 原始数据集
            analysis_results: 分析结果

        Returns:
            带标注的数据集
        """
        annotated_data = {}

        # 构建ID到难度的映射
        id_to_difficulty = {}
        for difficulty, samples in analysis_results.items():
            for ns in samples:
                id_to_difficulty[ns.id] = ns.difficulty

        for video_id, video_data in original_data.items():
            annotated_sts = []
            for item in video_data.get("sts", []):
                new_item = item.copy()
                if item.get("no_answer", False):
                    new_item["difficulty"] = id_to_difficulty.get(
                        item['id'],
                        'unknown'
                    )
                annotated_sts.append(new_item)

            annotated_data[video_id] = {
                "sts": annotated_sts,
                "duration": video_data["duration"]
            }

        return annotated_data
