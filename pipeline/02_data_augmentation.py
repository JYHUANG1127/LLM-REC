"""
==============================================================
 LLM-RecFusion — Stage 1: Sequence Generation & Data Augmentation
==============================================================
功能概述：
  1. 构建用户历史行为序列（按时间排序，截断/填充至固定长度）
  2. 数据增强：随机掩码（Masking）+ 随机裁剪（Cropping）
     → 提升模型对长尾和稀疏用户的鲁棒性
  3. 正负样本构造：用户下一个点击物品为正样本，随机采样未点击物品为负样本
  4. 时间轴拆分：每个用户按时间比例 80/10/10 切分训练/验证/测试集
  5. 数据落盘：分别保存 train/val/test 的 .parquet 文件

核心逻辑流：
  full_features.parquet
       │
       ▼
  build_sequences()  ──→ 按 user_id 分组，按 timestamp 排序
       │                  为每条样本构建 hist_movie_ids & hist_genres
       ▼
  split_sequences()  ──→ 每个用户内部按时间 80/10/10 切分
       │
       ├── Train ──→ apply_augmentation() ──→ generate_negative_samples()
       ├── Val   ──→ (no aug) ──→ generate_negative_samples()
       └── Test  ──→ (no aug) ──→ generate_negative_samples()
                           │
                           ▼
                    data/processed/{train,val,test}_data.parquet

内存优化：
  - build_sequences: 预分配 numpy int32 数组，避免 Python dict 中间表示
  - apply_augmentation: 全 numpy 向量化（Masking 矩阵级，Cropping 局部向量化）
  - generate_negative_samples: 逐行构建但仅处理正样本量的 4 倍
  - 中途 del 释放不再需要的 DataFrame
==============================================================
"""

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 0. 路径 & 超参数配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MAX_SEQ_LEN = 50          # 最大历史序列长度
MASK_RATIO = 0.15         # 随机掩码概率
CROP_PROB = 0.20          # 随机裁剪概率
NUM_NEGATIVES = 4         # 每条正样本对应的负样本数量
VAL_RATIO = 0.10          # 验证集比例
TEST_RATIO = 0.10         # 测试集比例
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 预定义列名（全局共用以减少重复构建）
HIST_MOVIE_COLS = [f"hist_movie_pos_{i}" for i in range(MAX_SEQ_LEN)]
HIST_G1_COLS = [f"hist_g1_pos_{i}" for i in range(MAX_SEQ_LEN)]
HIST_G2_COLS = [f"hist_g2_pos_{i}" for i in range(MAX_SEQ_LEN)]
HIST_G3_COLS = [f"hist_g3_pos_{i}" for i in range(MAX_SEQ_LEN)]
ALL_SEQ_COLS = HIST_MOVIE_COLS + HIST_G1_COLS + HIST_G2_COLS + HIST_G3_COLS


# ============================================================
# 1. 构建历史行为序列（预分配 numpy 数组，零 Python dict 开销）
# ============================================================
def build_sequences(
    df: pd.DataFrame,
    max_seq_len: int = MAX_SEQ_LEN,
) -> pd.DataFrame:
    """
    为每条样本生成用户在当前时间点之前的"历史行为序列"。

    性能优化：直接预分配 numpy int32 数组，填充后一次构建 DataFrame，
    避免 100 万行 × 205 列 Python dict 的内存爆炸。

    Parameters
    ----------
    df : pd.DataFrame
        全量特征宽表（已 encode + bucket）。
    max_seq_len : int
        最大序列长度，默认 50。

    Returns
    -------
    pd.DataFrame
        含 hist_movie_pos_*, hist_g*_pos_*, target_* 等列。
    """
    print("=" * 60)
    print("【1/5】构建用户历史行为序列（预分配 numpy 数组）...")
    print("=" * 60)

    df = df.sort_values(["user_id_encoded", "timestamp"]).reset_index(drop=True)

    # 预计算总输出行数 = sum(n_interactions_user - 1)
    user_n_interactions = df.groupby("user_id_encoded").size()
    total_rows = int((user_n_interactions - 1).clip(lower=0).sum())
    print(f"  ├─ 总样本数: {total_rows:,}")

    # 预分配 numpy 数组 (total_rows, ...)
    movie_arr = np.zeros((total_rows, max_seq_len), dtype=np.int32)
    g1_arr = np.zeros((total_rows, max_seq_len), dtype=np.int32)
    g2_arr = np.zeros((total_rows, max_seq_len), dtype=np.int32)
    g3_arr = np.zeros((total_rows, max_seq_len), dtype=np.int32)
    user_arr = np.zeros(total_rows, dtype=np.int32)
    target_movie_arr = np.zeros(total_rows, dtype=np.int32)
    target_g1_arr = np.zeros(total_rows, dtype=np.int32)
    target_g2_arr = np.zeros(total_rows, dtype=np.int32)
    target_g3_arr = np.zeros(total_rows, dtype=np.int32)

    # 逐用户填充
    row_idx = 0
    grouped = df.groupby("user_id_encoded")
    for user_id, group in grouped:
        group = group.reset_index(drop=True)
        n = len(group)

        m_ids = group["movie_id_encoded"].values.astype(np.int32)
        g1_v = group["genre_1"].values.astype(np.int32)
        g2_v = group["genre_2"].values.astype(np.int32)
        g3_v = group["genre_3"].values.astype(np.int32)

        for t in range(1, n):
            # 历史窗口 [max(0, t-max_seq_len), t)
            hist_start = max(0, t - max_seq_len)
            hist_len = t - hist_start
            pad_len = max_seq_len - hist_len

            # 写入用户和目标
            user_arr[row_idx] = user_id
            target_movie_arr[row_idx] = m_ids[t]
            target_g1_arr[row_idx] = g1_v[t]
            target_g2_arr[row_idx] = g2_v[t]
            target_g3_arr[row_idx] = g3_v[t]

            # 写入历史序列（直接 numpy 切片，零 Python 循环）
            if pad_len > 0:
                # 前 pad_len 位是 0（已在创建数组时初始化为 0）
                movie_arr[row_idx, pad_len:max_seq_len] = m_ids[hist_start:t]
                g1_arr[row_idx, pad_len:max_seq_len] = g1_v[hist_start:t]
                g2_arr[row_idx, pad_len:max_seq_len] = g2_v[hist_start:t]
                g3_arr[row_idx, pad_len:max_seq_len] = g3_v[hist_start:t]
            else:
                # 无 padding，直接复制最近的 max_seq_len 条
                movie_arr[row_idx, :] = m_ids[hist_start:t]
                g1_arr[row_idx, :] = g1_v[hist_start:t]
                g2_arr[row_idx, :] = g2_v[hist_start:t]
                g3_arr[row_idx, :] = g3_v[hist_start:t]

            row_idx += 1

    # 一次构建 DataFrame（numpy 数组 → 零复制）
    data = {
        "user_id_encoded": user_arr,
        "target_movie_id": target_movie_arr,
        "target_genre_1": target_g1_arr,
        "target_genre_2": target_g2_arr,
        "target_genre_3": target_g3_arr,
    }
    for i in range(max_seq_len):
        data[HIST_MOVIE_COLS[i]] = movie_arr[:, i]
        data[HIST_G1_COLS[i]] = g1_arr[:, i]
        data[HIST_G2_COLS[i]] = g2_arr[:, i]
        data[HIST_G3_COLS[i]] = g3_arr[:, i]

    result_df = pd.DataFrame(data)
    print(f"  ✓ 序列构建完成，共 {len(result_df):,} 条样本")
    print(f"     列数: {len(result_df.columns)}")
    print(f"     内存: {result_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    return result_df


# ============================================================
# 2. 数据增强（全 NumPy 向量化）
# ============================================================
def apply_augmentation(
    df: pd.DataFrame,
    max_seq_len: int = MAX_SEQ_LEN,
    mask_ratio: float = MASK_RATIO,
    crop_prob: float = CROP_PROB,
) -> pd.DataFrame:
    """
    对训练集执行数据增强。

    ==============================================================
    为什么数据增强能提升双塔模型应对长尾和稀疏用户的鲁棒性？
    ==============================================================
    1. 随机掩码（Masking）：
       - 现实世界中，用户行为日志存在大量噪声和缺失。用户可能因误点、
         缓存、隐私设置等原因，部分行为记录未被系统捕获。
       - 以 15% 概率将历史物品 ID 替换为 0（[MASK]），迫使模型在信息
         不完整时仍做出合理预测，提升对长尾物品的泛化能力。
       - 同时，Masking 是一种隐式正则化，防止模型过拟合到特定序列模式。

    2. 随机裁剪（Cropping）：
       - 推荐系统中大量用户属于"低活用户"或"新用户"（冷启动场景），
         行为历史极短。Cropping 人为制造"短序列"样本，使模型适应稀疏场景。
       - 解决了训练分布（长序列）vs 推理分布（短序列）不一致的问题，
         本质是一种数据分布再平衡（Distribution Re-balancing）。

    3. 综合效果：
       - 双塔模型 + DIN 的注意力机制对序列质量高度敏感。增强后的序列
         使每个 batch 的序列模式更多样，既缓解过拟合，又提升冷启动泛化。

    为什么只在训练集上做增强？
      - 验证集和测试集必须反映真实分布，增强会污染评估指标（数据泄露）。
    ==============================================================

    Parameters
    ----------
    df : pd.DataFrame
        训练集 DataFrame。
    max_seq_len : int
        序列长度。
    mask_ratio : float
        掩码概率，默认 0.15。
    crop_prob : float
        裁剪概率，默认 0.20。

    Returns
    -------
    pd.DataFrame
        增强后的 DataFrame。
    """
    print(f"  └─ 数据增强: mask_ratio={mask_ratio}, crop_prob={crop_prob}")

    # 提取 numpy 视图（不复制）
    movies = df[HIST_MOVIE_COLS].values.copy()
    g1 = df[HIST_G1_COLS].values.copy()
    g2 = df[HIST_G2_COLS].values.copy()
    g3 = df[HIST_G3_COLS].values.copy()
    n_rows = len(df)

    # -----------------------------------------------------------
    # Step A：随机裁剪（Cropping）— 仅选中的行局部向量化
    # -----------------------------------------------------------
    non_pad_counts = (movies != 0).sum(axis=1)
    crop_decisions = np.random.random(n_rows) < crop_prob
    valid = crop_decisions & (non_pad_counts > 1)
    crop_idx = np.where(valid)[0]

    for idx in crop_idx:
        npc = int(non_pad_counts[idx])
        max_crop = max(1, npc // 2)
        n_crop = np.random.randint(1, max_crop + 1)
        first = max_seq_len - npc
        rem = npc - n_crop

        movies[idx, first:first + rem] = movies[idx, first + n_crop:first + npc].copy()
        movies[idx, first + rem:first + npc] = 0
        g1[idx, first:first + rem] = g1[idx, first + n_crop:first + npc].copy()
        g1[idx, first + rem:first + npc] = 0
        g2[idx, first:first + rem] = g2[idx, first + n_crop:first + npc].copy()
        g2[idx, first + rem:first + npc] = 0
        g3[idx, first:first + rem] = g3[idx, first + n_crop:first + npc].copy()
        g3[idx, first + rem:first + npc] = 0

    print(f"     ├─ 裁剪: {len(crop_idx):,} 条 ({len(crop_idx)/max(1,n_rows):.1%})")

    # -----------------------------------------------------------
    # Step B：随机掩码（Masking）— 全矩阵向量化
    # -----------------------------------------------------------
    non_pad = movies != 0
    mask = non_pad & (np.random.random(movies.shape) < mask_ratio)
    movies[mask] = 0
    g1[mask] = 0
    g2[mask] = 0
    g3[mask] = 0
    masked_frac = mask.sum() / max(1, non_pad.sum())
    print(f"     └─ 掩码: {mask.sum():,} 个位置 ({masked_frac:.1%})")

    # 写回
    result_df = df.copy()
    for i in range(max_seq_len):
        result_df[HIST_MOVIE_COLS[i]] = movies[:, i]
        result_df[HIST_G1_COLS[i]] = g1[:, i]
        result_df[HIST_G2_COLS[i]] = g2[:, i]
        result_df[HIST_G3_COLS[i]] = g3[:, i]
    return result_df


# ============================================================
# 3. 正负样本构造
# ============================================================
def generate_negative_samples(
    df: pd.DataFrame,
    all_movie_ids: np.ndarray,
    user_clicked: dict[int, set[int]],
    n_neg: int = NUM_NEGATIVES,
) -> pd.DataFrame:
    """
    为每条正样本随机采样 n_neg 个用户未点击过的物品作为负样本。

    核心思想：
      - 正样本 = 用户下一个点击的物品（label=1）
      - 负样本 = 从用户未点击池中随机采样（label=0）
      - 每条正样本搭配 n_neg 个负样本，形成 1:N 训练对

    输出结构：
      - 1 行 label=1 + n_neg 行 label=0，所有行共享相同 user_id 和 hist_* 序列
    """
    print(f"  └─ 负采样: 每条正样本搭配 {n_neg} 个负样本")

    meta_cols = ["user_id_encoded"] + ALL_SEQ_COLS
    n_pos = len(df)

    # 预分配负样本数组
    n_neg_total = n_pos * n_neg
    neg_movies = np.zeros((n_neg_total, MAX_SEQ_LEN), dtype=np.int32)
    neg_g1 = np.zeros((n_neg_total, MAX_SEQ_LEN), dtype=np.int32)
    neg_g2 = np.zeros((n_neg_total, MAX_SEQ_LEN), dtype=np.int32)
    neg_g3 = np.zeros((n_neg_total, MAX_SEQ_LEN), dtype=np.int32)
    neg_users = np.zeros(n_neg_total, dtype=np.int32)
    neg_targets = np.zeros(n_neg_total, dtype=np.int32)
    neg_labels = np.zeros(n_neg_total, dtype=np.int32)

    neg_idx = 0
    for _, row in df.iterrows():
        uid = row["user_id_encoded"]
        clicked = user_clicked.get(uid, set())
        pool = [m for m in all_movie_ids if m not in clicked]
        sampled = random.sample(pool, min(n_neg, len(pool)))
        while len(sampled) < n_neg:
            sampled.append(0)

        for item_meta in sampled:
            neg_users[neg_idx] = uid
            neg_targets[neg_idx] = item_meta
            neg_labels[neg_idx] = 0
            # 复制历史序列
            for i in range(MAX_SEQ_LEN):
                neg_movies[neg_idx, i] = row[HIST_MOVIE_COLS[i]]
                neg_g1[neg_idx, i] = row[HIST_G1_COLS[i]]
                neg_g2[neg_idx, i] = row[HIST_G2_COLS[i]]
                neg_g3[neg_idx, i] = row[HIST_G3_COLS[i]]
            neg_idx += 1

    # 构建正样本 DataFrame
    pos_df = df.copy()
    pos_df["label"] = np.int32(1)

    # 构建负样本 DataFrame
    neg_data = {"user_id_encoded": neg_users, "label": neg_labels,
                "target_movie_id": neg_targets,
                "target_genre_1": np.zeros(n_neg_total, dtype=np.int32),
                "target_genre_2": np.zeros(n_neg_total, dtype=np.int32),
                "target_genre_3": np.zeros(n_neg_total, dtype=np.int32)}
    for i in range(MAX_SEQ_LEN):
        neg_data[HIST_MOVIE_COLS[i]] = neg_movies[:, i]
        neg_data[HIST_G1_COLS[i]] = neg_g1[:, i]
        neg_data[HIST_G2_COLS[i]] = neg_g2[:, i]
        neg_data[HIST_G3_COLS[i]] = neg_g3[:, i]
    neg_df = pd.DataFrame(neg_data)

    return pd.concat([pos_df, neg_df], ignore_index=True)


# ============================================================
# 4. 时间轴拆分
# ============================================================
def split_sequences(
    df: pd.DataFrame,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按每个用户内部的时间顺序，将样本拆分为训练/验证/测试集。

    为什么按用户内部分时间拆分？
      - 推荐系统必须遵循"时间因果律"：不能用未来数据预测过去。
      - 按用户内部时间轴拆分保证训练数据时间上早于验证和测试数据。
      - 全局随机拆分会导致数据泄露，评估指标过于乐观。
    """
    print("=" * 60)
    print("【2/5】按用户内部时间轴拆分训练/验证/测试集...")
    print("=" * 60)

    df = df.copy()
    df["_seq"] = df.groupby("user_id_encoded").cumcount()

    train_list, val_list, test_list = [], [], []

    for _, g in df.groupby("user_id_encoded"):
        g = g.sort_values("_seq").reset_index(drop=True)
        n = len(g)
        nt = max(1, int(np.ceil(n * test_ratio)))
        nv = max(1, int(np.ceil(n * val_ratio)))
        if n <= nt + nv:
            train_list.append(g)
        else:
            test_list.append(g.iloc[-nt:])
            val_list.append(g.iloc[-(nt + nv):-nt])
            train_list.append(g.iloc[:-(nt + nv)])

    def _concat(lst):
        return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()

    train_df = _concat(train_list)
    val_df = _concat(val_list)
    test_df = _concat(test_list)

    for d in [train_df, val_df, test_df]:
        if "_seq" in d.columns:
            d.drop(columns=["_seq"], inplace=True)

    print(f"  ✓ 训练集: {len(train_df):,} | 验证集: {len(val_df):,} | 测试集: {len(test_df):,}")
    return train_df, val_df, test_df


# ============================================================
# 5. 构建用户已点击物品集合
# ============================================================
def build_user_clicked_set(df: pd.DataFrame) -> dict[int, set[int]]:
    """构建 {user_id: set(movie_ids)} 用于负采样过滤。"""
    # 使用 groupby+apply 比逐行 iterrows 快得多
    return df.groupby("user_id_encoded")["movie_id_encoded"].apply(set).to_dict()


# ============================================================
# 6. 主流程
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("🚀 LLM-RecFusion — 用户行为序列构建与数据增强")
    print("=" * 60)

    # ----------------------------------------------------------
    # 6a. 读取
    # ----------------------------------------------------------
    print("\n【0/5】读取全量特征数据...")
    input_path = PROCESSED_DIR / "full_features.parquet"
    df = pd.read_parquet(input_path)
    for c in ["user_id_encoded", "movie_id_encoded", "genre_1", "genre_2", "genre_3"]:
        if c in df.columns:
            df[c] = df[c].astype(np.int32)
    print(f"  ✓ 形状: {df.shape}, 内存: {df.memory_usage(deep=True).sum()/1024**3:.2f} GB")

    # ----------------------------------------------------------
    # 6b. 用户已点击集合
    # ----------------------------------------------------------
    print("\n【预备】构建用户已点击物品集合...")
    user_clicked = build_user_clicked_set(df)
    all_movie_ids = df["movie_id_encoded"].unique()
    print(f"  ✓ {len(user_clicked):,} 用户, {len(all_movie_ids):,} 物品")

    # ----------------------------------------------------------
    # 6c. 构建序列
    # ----------------------------------------------------------
    seq_df = build_sequences(df, max_seq_len=MAX_SEQ_LEN)
    del df

    # ----------------------------------------------------------
    # 6d. 拆分
    # ----------------------------------------------------------
    train_raw, val_raw, test_raw = split_sequences(seq_df)
    del seq_df

    # ----------------------------------------------------------
    # 6e. 增强（仅训练集）
    # ----------------------------------------------------------
    print("\n【3/5】训练集数据增强（Masking + Cropping）...")
    train_aug = apply_augmentation(train_raw)
    del train_raw

    # ----------------------------------------------------------
    # 6f. 负采样
    # ----------------------------------------------------------
    print("\n【4/5】正负样本构造（负采样）...")
    print("  ▶ 训练集...")
    train_final = generate_negative_samples(train_aug, all_movie_ids, user_clicked)
    print(f"     训练集: {len(train_final):,} 条")
    del train_aug

    print("  ▶ 验证集...")
    val_final = generate_negative_samples(val_raw, all_movie_ids, user_clicked)
    print(f"     验证集: {len(val_final):,} 条")
    del val_raw

    print("  ▶ 测试集...")
    test_final = generate_negative_samples(test_raw, all_movie_ids, user_clicked)
    print(f"     测试集: {len(test_final):,} 条")
    del test_raw

    # ----------------------------------------------------------
    # 6g. 落盘
    # ----------------------------------------------------------
    print("\n【5/5】数据落盘...")
    output_cols = (["user_id_encoded"] + ALL_SEQ_COLS
                   + ["target_movie_id", "target_genre_1", "target_genre_2", "target_genre_3",
                      "label"])

    for fname, ds in [("train_data.parquet", train_final),
                      ("val_data.parquet", val_final),
                      ("test_data.parquet", test_final)]:
        out = PROCESSED_DIR / fname
        ds[[c for c in output_cols if c in ds.columns]].to_parquet(out, index=False)
        pos = (ds["label"] == 1).sum()
        neg = (ds["label"] == 0).sum()
        print(f"  ✓ {fname}: 形状 {ds.shape}, {out.stat().st_size/1024/1024:.1f} MB, "
              f"正 {pos:,} / 负 {neg:,} (1:{neg//max(1,pos)})")

    # ----------------------------------------------------------
    # 6h. 样本展示
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("📋 处理好的样本结构（训练集第 1 条正样本）:")
    print("=" * 60)
    sample = train_final[train_final["label"] == 1].iloc[0]
    for k in ["user_id_encoded", "target_movie_id", "target_genre_1",
              "target_genre_2", "target_genre_3", "label"]:
        print(f"  {k:30s}: {sample[k]}")
    non_pad = sum(1 for i in range(MAX_SEQ_LEN) if sample[f"hist_movie_pos_{i}"] != 0)
    print(f"  hist_movie_pos_[0..49]: {non_pad} 条非 padding")
    print(f"  hist_movie[:10]:        {[int(sample[f'hist_movie_pos_{i}']) for i in range(min(10, MAX_SEQ_LEN))]}")

    # ----------------------------------------------------------
    # 6i. 统计
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 训练集历史序列真实长度分布:")
    print("=" * 60)
    pos_df = train_final[train_final["label"] == 1]
    lens = (pos_df[HIST_MOVIE_COLS].values != 0).sum(axis=1)
    print(f"  均值: {lens.mean():.1f} | 中位数: {np.median(lens):.0f} | "
          f"最小: {lens.min()} | 最大: {lens.max()}")
    print(f"  短序列 (<5): {(lens < 5).mean():.2%}")
    print(f"  超长 (>=50): {(lens >= 50).mean():.2%}")

    print("\n" + "=" * 60)
    print("✅ 阶段一：序列构建与数据增强执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
