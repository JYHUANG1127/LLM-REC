"""
==========================================================================
 LLM-RecFusion — Stage 3: Coarse Ranking Data Pipeline (Data Pipeline)
==========================================================================
功能概述:
  为粗排模型（FM / Wide&Deep）构建高效、可复用的 PyTorch 数据输入管道。

核心设计原则:
  1. 全向量化数据转换：__init__ 中通过 NumPy/Pandas 向量化操作一次性完成
     所有类型转换，严禁逐行 for 循环遍历 DataFrame。
  2. 类型分离：离散特征 → torch.long（适配 Embedding 查找）
               连续特征 → torch.float32（适配梯度计算）
               标签     → torch.float32（适配二分类损失 BCEWithLogitsLoss）
  3. 零拷贝索引：__getitem__ 直接索引预转换的 Tensor，避免重复转换开销。

使用示例::
    >>> from datasets.coarse_ranking_dataset import CoarseRankingDataset
    >>> config = {
    ...     "sparse_features": ["user_id_encoded", "movie_id_encoded", "gender_encoded"],
    ...     "dense_features": ["age", "user_activity", "item_heat"],
    ...     "label": "rating",
    ...     "label_threshold": 3,       # rating > 3 → 正样本 1
    ... }
    >>> ds = CoarseRankingDataset(df, config)
    >>> features, label = ds[0]
    >>> features["sparse"].shape, features["dense"].shape, label.shape
==========================================================================
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CoarseRankingDataset(Dataset):
    """
    粗排模型专用 PyTorch Dataset。

    本类将 Pandas DataFrame 中的离散/连续特征批量转换为 Tensor，
    并支持灵活的标签构造（原始评分 → 二分类点击标签）。

    Parameters
    ----------
    data : pd.DataFrame
        包含离散特征、连续特征和标签列的宽表 DataFrame。
    feature_config : dict
        特征配置字典，需包含以下键：
        - "sparse_features": List[str]，离散特征列名列表
        - "dense_features": List[str]，连续特征列名列表
        - "label": str，标签列名
        - "label_threshold" (可选): float，用于将连续评分转为
          二分类标签（如 rating > threshold → 1），
          若不提供则直接使用原始值（适用于已二值化的标签列）。
    device : str, optional
        存储 Tensor 的设备（"cpu" / "cuda"）。
        默认 "cpu"，子类可在初始化后调用 .to(device) 整体迁移。

    Attributes
    ----------
    sparse_tensor : torch.LongTensor  [num_samples, num_sparse_features]
        离散特征张量，dtype=torch.long，供 Embedding 层直接使用。
    dense_tensor : torch.FloatTensor  [num_samples, num_dense_features]
        连续特征张量，dtype=torch.float32，供 DNN 层直接使用。
    labels : torch.FloatTensor  [num_samples]
        标签张量，dtype=torch.float32，适配 BCEWithLogitsLoss。
    num_samples : int
        数据集总样本数。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_config: Dict[str, Union[List[str], str, float]],
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # ── 1. 参数解析 ──
        self.sparse_features: List[str] = feature_config.get("sparse_features", [])
        self.dense_features: List[str] = feature_config.get("dense_features", [])
        self.label_col: str = feature_config.get("label", "rating")
        self.label_threshold: Optional[float] = feature_config.get("label_threshold", None)
        self.device: str = device

        # ── 2. 有效性校验 ──
        missing_cols = (
            set(self.sparse_features)
            | set(self.dense_features)
            | {self.label_col}
        ) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"DataFrame 中缺少以下列: {sorted(missing_cols)}"
            )

        self.num_samples: int = len(data)

        # ── 3. 离散特征转换：向量化 → torch.long ──
        #     提取所有离散特征列的 numpy int64 数组（形状 [N, S]），
        #     再整体转为 torch.long，零 for 循环。
        if self.sparse_features:
            sparse_np: np.ndarray = data[self.sparse_features].to_numpy(
                dtype=np.int64
            )
            self.sparse_tensor: torch.LongTensor = torch.as_tensor(
                sparse_np, dtype=torch.long, device=self.device
            )
        else:
            self.sparse_tensor = torch.empty(
                (self.num_samples, 0), dtype=torch.long, device=self.device
            )

        # ── 4. 连续特征转换：向量化 → torch.float32 ──
        #     提取所有连续特征列的 numpy float64 数组（形状 [N, D]），
        #     再整体转为 torch.float32。
        if self.dense_features:
            dense_np: np.ndarray = data[self.dense_features].to_numpy(
                dtype=np.float32
            )
            self.dense_tensor: torch.FloatTensor = torch.as_tensor(
                dense_np, dtype=torch.float32, device=self.device
            )
        else:
            self.dense_tensor = torch.empty(
                (self.num_samples, 0), dtype=torch.float32, device=self.device
            )

        # ── 5. 标签转换：向量化 → torch.float32 ──
        #     若指定了 label_threshold，则将连续评分转为二分类标签；
        #     否则直接使用原始值（要求已是 0/1 格式）。
        labels_np: np.ndarray = data[self.label_col].to_numpy(dtype=np.float32)
        if self.label_threshold is not None:
            labels_np = (labels_np > self.label_threshold).astype(np.float32)
        self.labels: torch.FloatTensor = torch.as_tensor(
            labels_np, dtype=torch.float32, device=self.device
        )

        # ── 6. 日志输出 ──
        n_sparse = len(self.sparse_features)
        n_dense = len(self.dense_features)
        pos_ratio = self.labels.float().mean().item()
        print(
            f"[CoarseRankingDataset] 初始化完成: "
            f"{self.num_samples:,} 样本, "
            f"{n_sparse} 离散特征 + {n_dense} 连续特征, "
            f"正样本比例 {pos_ratio:.2%}"
        )

    def __len__(self) -> int:
        """
        返回数据集总样本数。

        Returns
        -------
        int
            数据集中的样本数量。
        """
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        获取单个样本的特征字典与标签。

        Parameters
        ----------
        idx : int
            样本索引，取值范围 [0, num_samples)。

        Returns
        -------
        features : Dict[str, torch.Tensor]
            包含两个键值对的字典：
            - "sparse": torch.LongTensor  [num_sparse_features]
            - "dense":  torch.FloatTensor  [num_dense_features]
        label : torch.Tensor
            标量 Tensor，dtype=torch.float32。
        """
        features: Dict[str, torch.Tensor] = {
            "sparse": self.sparse_tensor[idx],
            "dense": self.dense_tensor[idx],
        }
        return features, self.labels[idx]

    def to(self, device: str) -> "CoarseRankingDataset":
        """
        将 Dataset 中所有预转换的 Tensor 迁移到指定设备。

        支持链式调用::

            ds = CoarseRankingDataset(df, config).to("cuda:0")

        Parameters
        ----------
        device : str
            目标设备，如 "cuda:0" 或 "cpu"。
        """
        self.sparse_tensor = self.sparse_tensor.to(device)
        self.dense_tensor = self.dense_tensor.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

    def get_feature_dims(self) -> Dict[str, Union[int, List[int]]]:
        """
        返回特征维度信息，供模型构建时动态确定各层输入尺寸。

        Returns
        -------
        dict
            - "num_sparse": int，离散特征数量
            - "num_dense": int，连续特征数量
            - "sparse_feature_names": List[str]
            - "dense_feature_names": List[str]
        """
        return {
            "num_sparse": len(self.sparse_features),
            "num_dense": len(self.dense_features),
            "sparse_feature_names": self.sparse_features,
            "dense_feature_names": self.dense_features,
        }
