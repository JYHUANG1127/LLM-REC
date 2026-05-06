"""
==============================================================
 LLM-RecFusion — Stage 1: Feature Engineering & Bucketing
==============================================================
功能概述：
  1. 多表关联：将 movies.dat、users.dat、ratings.dat 关联为一张大宽表
  2. 离散特征 Label Encoding：将 user_id, movie_id, gender, occupation
     映射为连续整数 ID，并统计 Vocab Size 写入 feature_config.json
  3. 连续特征等频分桶：构造 Item Heat（电影被评价次数）和 User Activity
     （用户评价次数），并做 Quantile Bucketing → 10 个桶
  4. 多值特征处理：genres 按 "|" 切分 → 固定长度 max_len=3 的 ID 序列
  5. 数据落盘：保存为 data/processed/full_features.parquet

输出：
  - data/processed/full_features.parquet   ← 全量特征宽表
  - data/processed/feature_config.json      ← 特征配置（含 Vocab Size）
==============================================================
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 0. 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-1m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 多表关联 —— 读取原始数据并合并为一张大宽表
# ============================================================
print("=" * 60)
print("【1/5】正在读取原始数据并执行多表关联...")
print("=" * 60)

# 1a. 评分数据：UserID::MovieID::Rating::Timestamp
ratings = pd.read_csv(
    RAW_DIR / "ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
)
print(f"  ✓ ratings.dat 加载完成，共 {len(ratings):,} 条评分记录")

# 1b. 用户数据：UserID::Gender::Age::Occupation::Zip-code
users = pd.read_csv(
    RAW_DIR / "users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip_code"],
)
print(f"  ✓ users.dat 加载完成，共 {len(users):,} 条用户记录")

# 1c. 电影数据：MovieID::Title::Genres
movies = pd.read_csv(
    RAW_DIR / "movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1",
)
print(f"  ✓ movies.dat 加载完成，共 {len(movies):,} 条电影记录")

# 1d. 合并：ratings ← users (on user_id) ← movies (on movie_id)
df = ratings.merge(users, on="user_id", how="left").merge(
    movies, on="movie_id", how="left"
)
print(f"  ✓ 多表关联完成，宽表形状: {df.shape}")

# ============================================================
# 2. 离散特征 Label Encoding & Vocab Size 统计
# ============================================================
print("\n" + "=" * 60)
print("【2/5】离散特征 Label Encoding & Vocab Size 统计...")
print("=" * 60)


def label_encode(series: pd.Series) -> tuple[pd.Series, int]:
    """
    对离散特征执行 Label Encoding。

    Parameters
    ----------
    series : pd.Series
        待编码的原始离散特征列。

    Returns
    -------
    encoded : pd.Series
        编码后的整数序列（0 ~ vocab_size-1）。
    vocab_size : int
        词表大小，即唯一取值个数。
    """
    uniques = series.unique()
    # 构建映射字典，确保 0 不被占用（0 留给 OOV/填充）
    mapping = {val: idx + 1 for idx, val in enumerate(uniques)}
    encoded = series.map(mapping)
    # vocab_size = 唯一值个数（不含 0），但我们在 Embedding 层使用
    # vocab_size + 1 作为词表大小（含填充位 0）
    vocab_size = len(uniques)
    return encoded, vocab_size


# 执行 Label Encoding
discrete_features = {
    "user_id": df["user_id"],
    "movie_id": df["movie_id"],
    "gender": df["gender"],
    "occupation": df["occupation"],
}

feature_config = {}
encoded_cols = {}
for col_name, series in discrete_features.items():
    encoded, vocab_size = label_encode(series)
    encoded_cols[f"{col_name}_encoded"] = encoded
    feature_config[col_name] = {
        "vocab_size": vocab_size + 1,  # +1 为 padding/OOV 预留 0
        "embedding_dim": None,  # 后续双塔模型动态确定
    }
    print(
        f"  ✓ {col_name}: 编码完成，"
        f"Vocab Size = {vocab_size + 1:,} "
        f"(含 padding 位)"
    )

# 将编码列加入宽表
for col_name, encoded in encoded_cols.items():
    df[col_name] = encoded

# ============================================================
# 3. 连续特征构建 & 等频分桶（Quantile Bucketing）
# ============================================================
print("\n" + "=" * 60)
print("【3/5】连续特征构建 & 等频分桶（Quantile Bucketing）...")
print("=" * 60)

# 3a. Item Heat：每部电影的历史被评价次数
item_heat = df.groupby("movie_id")["rating"].count().rename("item_heat")
df = df.merge(item_heat, on="movie_id", how="left")
print(f"  ✓ Item Heat 特征构建完成")
print(f"     最小值: {df['item_heat'].min()}, "
      f"最大值: {df['item_heat'].max()}, "
      f"中位数: {df['item_heat'].median():.0f}")

# 3b. User Activity：每个用户的历史评价次数
user_activity = df.groupby("user_id")["rating"].count().rename("user_activity")
df = df.merge(user_activity, on="user_id", how="left")
print(f"  ✓ User Activity 特征构建完成")
print(f"     最小值: {df['user_activity'].min()}, "
      f"最大值: {df['user_activity'].max()}, "
      f"中位数: {df['user_activity'].median():.0f}")


def quantile_bucket(
    series: pd.Series, n_buckets: int = 10
) -> pd.Series:
    """
    等频分桶（Quantile Bucketing）：将连续值按分位数划分为 n_buckets 个桶，
    使每个桶内的样本数尽可能相等。

    算法步骤：
      1. 使用 pd.qcut 基于分位数将连续值切分为 n_buckets 个区间。
      2. 当存在大量重复值时，pd.qcut 可能因分位数边界重复而报错，
         此时回退到 rank-based 分桶：按排序后的秩等分。
      3. 桶标签从 1 开始（区别于 0-padding），便于 Embedding 层使用。

    Parameters
    ----------
    series : pd.Series
        待分桶的连续特征列。
    n_buckets : int
        桶的数量，默认 10。

    Returns
    -------
    bucketed : pd.Series
        分桶后的离散标签（1 ~ n_buckets）。
    """
    try:
        # 优先使用 pd.qcut 进行等频分桶
        # labels=False 返回整数索引（0 ~ n_buckets-1），我们 +1 偏移到 1 ~ n_buckets
        bucketed = (
            pd.qcut(series, q=n_buckets, duplicates="drop", labels=False)
            + 1
        )
    except ValueError:
        # 容错：当 qcut 无法处理时（如数据过于集中），使用 rank 分桶
        print("      ⚠ pd.qcut 失败，回退到 rank-based 分桶")
        ranks = series.rank(method="first")
        bucketed = (
            pd.cut(
                ranks,
                bins=n_buckets,
                labels=False,
            )
            + 1
        )
    return bucketed.astype(np.int32)


# 对 Item Heat 进行等频分桶
df["item_heat_bucket"] = quantile_bucket(df["item_heat"], n_buckets=10)
print(f"  ✓ Item Heat 等频分桶完成（10 个桶）")
print(f"     桶分布:\n{df['item_heat_bucket'].value_counts().sort_index().to_string()}")

# 对 User Activity 进行等频分桶
df["user_activity_bucket"] = quantile_bucket(df["user_activity"], n_buckets=10)
print(f"\n  ✓ User Activity 等频分桶完成（10 个桶）")
print(f"     桶分布:\n{df['user_activity_bucket'].value_counts().sort_index().to_string()}")

# 将分桶信息写入 feature_config
feature_config["item_heat_bucket"] = {
    "vocab_size": 11,  # 10 个桶 + padding 位 0
    "embedding_dim": None,
    "bucketing": "quantile",
    "n_buckets": 10,
}
feature_config["user_activity_bucket"] = {
    "vocab_size": 11,
    "embedding_dim": None,
    "bucketing": "quantile",
    "n_buckets": 10,
}

# ============================================================
# 4. 多值特征处理 —— genres 按 "|" 切分，构建定长 ID 序列
# ============================================================
print("\n" + "=" * 60)
print("【4/5】多值特征处理 —— genres 切分与定长化...")
print("=" * 60)

# 4a. 提取 genres 中所有唯一的子类别，构建词表
all_genres = df["genres"].str.split("|").explode().unique()
genre_vocab = {g: idx + 1 for idx, g in enumerate(all_genres)}  # 0 留给 padding
genre_vocab_size = len(all_genres)
print(f"  ✓ genres 词表构建完成，共 {genre_vocab_size} 个唯一类型")
print(f"     类型列表: {list(genre_vocab.keys())}")

MAX_GENRE_LEN = 3  # 固定序列长度


def genres_to_fixed_sequence(genres_str: str, max_len: int = MAX_GENRE_LEN) -> list[int]:
    """
    将 genres 字符串（如 "Action|Sci-Fi|Comedy"）转化为固定长度的 ID 序列。

    处理逻辑：
      1. 按 "|" 切分字符串，得到子类别列表。
      2. 将每个子类别映射为词表中的 ID。
      3. 如果长度超过 max_len，截断；如果长度不足，在末尾补 0。

    Parameters
    ----------
    genres_str : str
        原始的 genres 字符串，如 "Action|Sci-Fi|Comedy"。
    max_len : int
        输出序列的固定长度。

    Returns
    -------
    list[int]
        固定长度的 ID 序列，如 [3, 7, 1]。
    """
    tokens = genres_str.split("|")
    ids = [genre_vocab.get(t, 0) for t in tokens]
    # 截断或填充至固定长度
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return ids


# 应用转换，展开为多列
genre_sequences = df["genres"].apply(genres_to_fixed_sequence)
genre_df = pd.DataFrame(
    genre_sequences.tolist(),
    columns=[f"genre_{i+1}" for i in range(MAX_GENRE_LEN)],
    index=df.index,
)
df = pd.concat([df, genre_df], axis=1)
print(f"  ✓ genres 多值特征处理完成")
print(f"     前 5 条样例:")
for i in range(min(5, len(df))):
    print(f"       {df.loc[i, 'genres']:40s} → [{', '.join(map(str, genre_sequences[i]))}]")

# 将 genre 特征信息写入 feature_config
feature_config["genres"] = {
    "vocab_size": genre_vocab_size + 1,  # +1 为 padding 位预留
    "embedding_dim": None,
    "max_sequence_length": MAX_GENRE_LEN,
    "tokenizer": "split_by_pipe",
}

# ============================================================
# 5. 数据落盘 & 输出验证
# ============================================================
print("\n" + "=" * 60)
print("【5/5】数据落盘 & 输出验证...")
print("=" * 60)

# 5a. 选取最终输出列（按推荐系统特征工程规范排序）
output_columns = [
    # --- 用户侧特征 ---
    "user_id_encoded",  # 用户 ID (离散)
    "gender_encoded",   # 性别 (离散)
    "occupation_encoded",  # 职业 (离散)
    "age",              # 年龄 (原始连续值，供后续交叉特征使用)
    "user_activity",    # 用户活跃度 (连续)
    "user_activity_bucket",  # 用户活跃度分桶 (离散)
    # --- 物品侧特征 ---
    "movie_id_encoded",  # 电影 ID (离散)
    "item_heat",        # 物品热度 (连续)
    "item_heat_bucket",  # 物品热度分桶 (离散)
    "genre_1",          # 类型 1 (多值离散)
    "genre_2",          # 类型 2 (多值离散)
    "genre_3",          # 类型 3 (多值离散)
    # --- 标签侧 ---
    "rating",           # 评分 (Label)
    # --- 辅助/原始字段（供调试/回查） ---
    "user_id",
    "movie_id",
    "title",
    "genres",
    "gender",
    "occupation",
    "timestamp",
]

# 确保所有列都存在，只保留存在的列
available_columns = [col for col in output_columns if col in df.columns]
df_final = df[available_columns].copy()

# 5b. 保存为 Parquet 格式
parquet_path = PROCESSED_DIR / "full_features.parquet"
df_final.to_parquet(parquet_path, index=False)
print(f"  ✓ 全量特征数据已保存至: {parquet_path}")
print(f"     文件大小: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")
print(f"     数据形状: {df_final.shape}")

# 5c. 保存 feature_config.json
config_path = PROCESSED_DIR / "feature_config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=2, ensure_ascii=False)
print(f"  ✓ 特征配置文件已保存至: {config_path}")

# ============================================================
# 6. 控制台输出验证
# ============================================================
print("\n" + "=" * 60)
print("📋 特征数据前 5 行样本:")
print("=" * 60)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)
print(df_final.head(5).to_string())

print("\n" + "=" * 60)
print("📋 feature_config.json 概览:")
print("=" * 60)
print(json.dumps(feature_config, indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
print("✅ 特征工程全流程执行完毕！")
print("=" * 60)
