"""
===============================================================
 LLM-RecFusion — Shared Data Pipeline: Global Data Consistency
===============================================================
工业级数据流水线。提供原子化的数据加载、划分与持久化功能，
确保 ALS 召回与 LLM 双塔召回使用完全相同的 Train/Test 数据快照。

核心流程:
  1. 自适应路径嗅探: 直捣 data/raw/ 读取最原始的 MovieLens-1M 数据
  2. 按时间序列划分: UserID + Timestamp 排序 → Leave-One-Out (或按比例)
  3. 持久化共享快照: 将划分结果存为中间文件，给 ALS / LLM 模块共享

Usage:
    from pipeline.data_pipeline import (
        sniff_raw_data_path,
        load_ratings,
        load_movies,
        chronological_split,
        build_als_interaction_matrix,
        build_test_ground_truth,
    )
"""

import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")


# ================================================================
# 1. 项目根目录自动嗅探
# ================================================================

def find_project_root() -> Path:
    """
    自适应嗅探项目根目录。
    支持从 notebooks/ 子目录或项目根目录直接启动。
    检测标志: data/raw/ml-1m/ratings.dat 是否存在。
    """
    candidate = Path.cwd()
    # 向上搜索最多 5 层
    for _ in range(5):
        if (candidate / "data" / "raw" / "ml-1m" / "ratings.dat").exists():
            return candidate
        if (candidate / "raw" / "ml-1m" / "ratings.dat").exists():
            return candidate
        candidate = candidate.parent

    # fallback: 当前目录
    return Path.cwd()


# ================================================================
# 2. 自适应路径嗅探器
# ================================================================

def sniff_raw_data_path(
    base_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    自适应嗅探 data/raw/ 下的 MovieLens-1M 原始数据路径。

    支持的分隔符:
      - .dat 文件: :: 分隔符 (标准 MovieLens 格式)
      - .csv 文件: , 分隔符

    Returns
    -------
    raw_dir : Path  -> data/raw/ml-1m 目录
    ratings_path : Path -> ratings.dat 或 ratings.csv 路径
    movies_path : Path   -> movies.dat 或 movies.csv 路径
    """
    if base_dir is None:
        base_dir = find_project_root()

    # 候选路径列表 (按优先级)
    candidates_ratings = [
        base_dir / "data" / "raw" / "ml-1m" / "ratings.dat",
        base_dir / "data" / "raw" / "ml-1m" / "ratings.csv",
        base_dir / "raw" / "ml-1m" / "ratings.dat",
        base_dir / "data" / "raw" / "ratings.dat",
        base_dir / "data" / "raw" / "ratings.csv",
    ]
    candidates_movies = [
        base_dir / "data" / "raw" / "ml-1m" / "movies.dat",
        base_dir / "data" / "raw" / "ml-1m" / "movies.csv",
        base_dir / "raw" / "ml-1m" / "movies.dat",
        base_dir / "data" / "raw" / "movies.dat",
        base_dir / "data" / "raw" / "movies.csv",
    ]

    ratings_path: Optional[Path] = None
    movies_path: Optional[Path] = None
    raw_dir: Optional[Path] = None

    for p in candidates_ratings:
        if p.exists():
            ratings_path = p
            raw_dir = p.parent
            break

    for p in candidates_movies:
        if p.exists():
            movies_path = p
            break

    if ratings_path is None:
        raise FileNotFoundError(
            f"❌ 无法找到 ratings 原始数据！搜索路径:\n"
            + "\n".join(f"  - {p}" for p in candidates_ratings)
        )
    if movies_path is None:
        raise FileNotFoundError(
            f"❌ 无法找到 movies 原始数据！搜索路径:\n"
            + "\n".join(f"  - {p}" for p in candidates_movies)
        )

    return raw_dir, ratings_path, movies_path


# ================================================================
# 3. 加载原始评分数据
# ================================================================

def load_ratings(
    ratings_path: Path,
) -> pd.DataFrame:
    """
    加载 MovieLens-1M ratings 原始数据。

    自动检测分隔符:
      - .dat → '::'
      - .csv → ','

    Returns
    -------
    DataFrame with columns: UserID, MovieID, Rating, Timestamp
    数据类型已优化 (int32, float32, int64)
    """
    suffix = ratings_path.suffix.lower()

    if suffix == ".dat":
        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            dtype={
                "UserID": np.int32,
                "MovieID": np.int32,
                "Rating": np.float32,
                "Timestamp": np.int64,
            },
        )
    elif suffix == ".csv":
        df = pd.read_csv(
            ratings_path,
            dtype={
                "UserID": np.int32,
                "MovieID": np.int32,
                "Rating": np.float32,
                "Timestamp": np.int64,
            },
        )
        # 确保列名统一
        df.columns = [c.capitalize() for c in df.columns]
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .dat 和 .csv")

    return df


# ================================================================
# 4. 加载电影元数据
# ================================================================

def load_movies(
    movies_path: Path,
) -> pd.DataFrame:
    """
    加载 MovieLens-1M movies 原始数据。

    Returns
    -------
    DataFrame with columns: MovieID, Title, Genres
    """
    suffix = movies_path.suffix.lower()

    if suffix == ".dat":
        df = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["MovieID", "Title", "Genres"],
            dtype={"MovieID": np.int32, "Title": str, "Genres": str},
            encoding="latin-1",
        )
    elif suffix == ".csv":
        df = pd.read_csv(movies_path, encoding="latin-1")
        df.columns = [c.capitalize() for c in df.columns]
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .dat 和 .csv")

    return df


# ================================================================
# 5. 严格按时间序列划分 (Chronological Split)
# ================================================================

def chronological_split(
    ratings: pd.DataFrame,
    strategy: str = "loo",
    n_test: int = 1,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    严格按时间序列 (UserID + Timestamp) 划分训练集和测试集。

    支持两种划分策略:
      1. Leave-One-Out (loo):
         每个用户时间戳最晚的 n_test 条交互作为测试集。
         推荐 n_test=1 (经典 LOO), n_test=5 (更充分的评估)
      2. 按比例划分 (ratio):
         每个用户前 (1 - test_ratio) 的交互作为训练集，
         最后 test_ratio 的交互作为测试集。

    Parameters
    ----------
    ratings : pd.DataFrame
        包含 UserID, MovieID, Rating, Timestamp 列的 DataFrame。
    strategy : str, default="loo"
        划分策略: "loo" | "ratio"
    n_test : int, default=1
        (仅 loo 策略) 每个用户最后 N 条交互划入测试集。
    test_ratio : float, default=0.2
        (仅 ratio 策略) 每个用户最后 test_ratio 比例划入测试集。

    Returns
    -------
    train_df : pd.DataFrame
        训练集 (历史记忆)
    test_df : pd.DataFrame
        测试集 (Ground Truth)
    """
    print("=" * 60)
    print("  Chronological Split — 严格按时间序列划分数据")
    print("=" * 60)

    # 按 UserID + Timestamp 排序 (时间戳升序，历史行为在前)
    ratings = ratings.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)

    print(f"  总评分记录: {len(ratings):,}")
    print(f"  唯一用户:   {ratings['UserID'].nunique():,}")
    print(f"  唯一电影:   {ratings['MovieID'].nunique():,}")
    print(f"  时间戳范围: [{ratings['Timestamp'].min()}, {ratings['Timestamp'].max()}]")

    # 为每个用户计算交互序列位置
    ratings["user_seq"] = ratings.groupby("UserID").cumcount() + 1
    ratings["user_total"] = ratings.groupby("UserID")["user_seq"].transform("max")

    if strategy == "loo":
        # Leave-One-Out: 每个用户最后 n_test 条为测试集
        train_mask = ratings["user_seq"] <= (ratings["user_total"] - n_test)
        strategy_desc = f"Leave-{n_test}-Out"
    elif strategy == "ratio":
        # 按比例: 每个用户前 (1-test_ratio) 为训练集
        train_mask = ratings["user_seq"] <= (ratings["user_total"] * (1 - test_ratio))
        strategy_desc = f"Ratio ({1-test_ratio:.0%} / {test_ratio:.0%})"
    else:
        raise ValueError(f"strategy 必须是 'loo' 或 'ratio'，收到: {strategy}")

    test_mask = ~train_mask

    train_df = ratings[train_mask].copy().drop(columns=["user_seq", "user_total"])
    test_df = ratings[test_mask].copy().drop(columns=["user_seq", "user_total"])

    print(f"\n  策略: {strategy_desc}")
    print(f"  ════════════════════════════════════════")
    print(f"  训练集记录: {len(train_df):>10,}  ({len(train_df)/len(ratings)*100:.1f}%)")
    print(f"  测试集记录: {len(test_df):>10,}  ({len(test_df)/len(ratings)*100:.1f}%)")
    print(f"  ════════════════════════════════════════")
    print(f"  训练集唯一用户: {train_df['UserID'].nunique():,}")
    print(f"  训练集唯一电影: {train_df['MovieID'].nunique():,}")
    print(f"  测试集唯一用户: {test_df['UserID'].nunique():,}")
    print(f"  测试集唯一电影: {test_df['MovieID'].nunique():,}")

    # 验证: 测试集中的用户都出现在训练集中
    missing_users = set(test_df["UserID"].unique()) - set(train_df["UserID"].unique())
    if missing_users:
        print(f"\n  ⚠️  警告: {len(missing_users)} 个测试用户不在训练集中")

    return train_df, test_df


# ================================================================
# 6. 构建 ALS 交互矩阵 (CSR)
# ================================================================

def build_als_interaction_matrix(
    train_df: pd.DataFrame,
) -> Tuple[csr_matrix, dict, dict, dict]:
    """
    从训练集构建 User-Item 稀疏矩阵 (CSR)，用于 ALS 模型训练。

    使用隐式反馈: 评分 → 二元交互 (value=1.0)

    Parameters
    ----------
    train_df : pd.DataFrame
        训练集，包含 UserID, MovieID 列。

    Returns
    -------
    interaction_matrix : csr_matrix
        User-Item 稀疏矩阵，形状 [n_users, n_items]
    user2idx : dict
        原始 UserID → 0-based 索引
    item2idx : dict
        原始 MovieID → 0-based 索引
    idx2item : dict
        0-based 索引 → 原始 MovieID
    """
    print("=" * 60)
    print("  Building ALS Interaction Matrix (CSR)...")
    print("=" * 60)

    train_users = sorted(train_df["UserID"].unique())
    train_items = sorted(train_df["MovieID"].unique())

    user2idx = {uid: i for i, uid in enumerate(train_users)}
    item2idx = {iid: i for i, iid in enumerate(train_items)}
    idx2item = {i: iid for iid, i in item2idx.items()}

    n_users = len(train_users)
    n_items = len(train_items)

    print(f"  训练用户数: {n_users:,}")
    print(f"  训练物品数: {n_items:,}")

    # 隐式反馈: 所有交互赋值为 1.0
    rows = train_df["UserID"].map(user2idx).values.astype(np.int32)
    cols = train_df["MovieID"].map(item2idx).values.astype(np.int32)
    data = np.ones(len(train_df), dtype=np.float64)

    interaction_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float64,
    )

    print(f"  CSR 矩阵形状: {interaction_matrix.shape}")
    print(f"  非零元素数:   {interaction_matrix.nnz:,}")
    sparsity = 1 - interaction_matrix.nnz / (n_users * n_items)
    print(f"  稀疏度:       {sparsity:.6%}")

    return interaction_matrix, user2idx, item2idx, idx2item


# ================================================================
# 7. 构建测试集 Ground Truth
# ================================================================

def build_test_ground_truth(
    test_df: pd.DataFrame,
    user2idx: Optional[dict] = None,
) -> dict:
    """
    构建测试集 Ground Truth 字典。

    Parameters
    ----------
    test_df : pd.DataFrame
        测试集，包含 UserID, MovieID 列。
    user2idx : dict, optional
        原始 UserID 到索引的映射。如果提供，只保留训练集中存在的用户。

    Returns
    -------
    test_user_groups : dict
        {user_id: set(movie_ids)}
    """
    test_user_groups = {}
    for uid, group in test_df.groupby("UserID"):
        if user2idx is None or uid in user2idx:
            test_user_groups[int(uid)] = set(group["MovieID"].values)

    print(f"\n  Ground Truth 用户数: {len(test_user_groups):,}")
    if user2idx is not None:
        print(f"  (仅保留训练集中出现过的用户)")

    return test_user_groups


# ================================================================
# 8. 持久化共享数据快照
# ================================================================

def save_data_snapshot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies: pd.DataFrame,
    save_dir: Path,
    prefix: str = "ml_1m",
) -> None:
    """
    保存数据划分快照到磁盘，确保 ALS 和 LLM 模块使用完全相同的数据。

    保存文件:
      - {save_dir}/{prefix}_train.parquet
      - {save_dir}/{prefix}_test.parquet
      - {save_dir}/{prefix}_movies.parquet
      - {save_dir}/{prefix}_split_info.json
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_path = save_dir / f"{prefix}_train.parquet"
    test_path = save_dir / f"{prefix}_test.parquet"
    movies_path = save_dir / f"{prefix}_movies.parquet"
    info_path = save_dir / f"{prefix}_split_info.json"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    movies.to_parquet(movies_path, index=False)

    info = {
        "n_train_records": len(train_df),
        "n_test_records": len(test_df),
        "n_train_users": int(train_df["UserID"].nunique()),
        "n_test_users": int(test_df["UserID"].nunique()),
        "n_train_items": int(train_df["MovieID"].nunique()),
        "n_test_items": int(test_df["MovieID"].nunique()),
        "n_movies_total": len(movies),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\n  ✅ 数据快照已保存至: {save_dir}")
    print(f"     - 训练集: {train_path.name} ({len(train_df):,} 条)")
    print(f"     - 测试集: {test_path.name} ({len(test_df):,} 条)")
    print(f"     - 电影:   {movies_path.name} ({len(movies):,} 部)")
    print(f"     - 信息:   {info_path.name}")


def load_data_snapshot(
    save_dir: Path,
    prefix: str = "ml_1m",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从磁盘加载数据划分快照。
    """
    save_dir = Path(save_dir)

    train_path = save_dir / f"{prefix}_train.parquet"
    test_path = save_dir / f"{prefix}_test.parquet"
    movies_path = save_dir / f"{prefix}_movies.parquet"

    if not all(p.exists() for p in [train_path, test_path, movies_path]):
        raise FileNotFoundError(
            f"数据快照不完整，请先运行 save_data_snapshot() 生成快照。\n"
            f"缺少文件: {[p for p in [train_path, test_path, movies_path] if not p.exists()]}"
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    movies = pd.read_parquet(movies_path)

    print(f"  ✅ 数据快照加载成功:")
    print(f"     - 训练集: {train_path.name} ({len(train_df):,} 条)")
    print(f"     - 测试集: {test_path.name} ({len(test_df):,} 条)")
    print(f"     - 电影:   {movies_path.name} ({len(movies):,} 部)")

    return train_df, test_df, movies


# ================================================================
# 9. 统一保存召回结果
# ================================================================

def save_recall_results(
    recall_dict: dict,
    output_path: Path,
    model_name: str,
    top_k: int,
) -> None:
    """
    保存召回结果并打印 JSON 文件信息。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recall_dict, f, ensure_ascii=False)

    file_size_kb = output_path.stat().st_size / 1024

    print(f"  ✅ {model_name} 召回结果已保存: {output_path}")
    print(f"  📦 文件大小: {file_size_kb:.2f} KB")
    print(f"  👤 覆盖用户数: {len(recall_dict):,}")
    print(f"  🎯 Top-K: {top_k}")

    # 预览第一个用户
    if recall_dict:
        sample_user = list(recall_dict.keys())[0]
        print(f"  📝 预览 (用户 {sample_user} 的 Top-{min(5, top_k)}):")
        print(f"    {recall_dict[sample_user][:min(5, top_k)]}")


def compute_recall_at_k(
    recall_dict: dict,
    test_user_groups: dict,
    top_k: int,
) -> float:
    """
    计算 Recall@K 评估指标。

    Returns
    -------
    mean_recall : float
        所有用户的平均 Recall@K
    """
    recall_list = []
    for uid, gt_set in test_user_groups.items():
        if uid not in recall_dict:
            continue
        rec_set = set(recall_dict[uid])
        hits = len(rec_set & gt_set)
        recall_list.append(hits / len(gt_set) if len(gt_set) > 0 else 0.0)

    mean_recall = float(np.mean(recall_list)) if recall_list else 0.0
    n_users = len(recall_list)

    print(f"\n  🔹 Recall@{top_k} (均值): {mean_recall:.4f} ({mean_recall*100:.2f}%)")
    print(f"  📊 评估用户数: {n_users:,}")

    return mean_recall
