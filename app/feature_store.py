"""
LLM-RecFusion — 在线特征存储与批量查询引擎

对接 Redis 底层特征存储，为召回/排序阶段提供用户画像、
物品属性的极速读取能力。所有查询均采用异步非阻塞 I/O，
避免 GIL 和磁盘 IO 带来的延迟抖动。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# ───────────────────── Redis Key 前缀模板 ──────────────────────────────
USER_FEATURE_KEY = "feature:user:{user_id}"
ITEM_FEATURE_KEY = "feature:item:{item_id}"

# ───────────────────── 默认特征（Cache Miss 时返回） ─────────────────────
_DEFAULT_USER_FEATURES: Dict[str, Any] = {
    "user_id": 0,
    "age": 25,
    "gender": "unknown",
    "city": "unknown",
    "active_level": 1.0,
}

_DEFAULT_ITEM_FEATURES: Dict[str, Any] = {
    "item_id": 0,
    "category": "unknown",
    "sub_category": "unknown",
    "price": 0.0,
    "title": "",
}


class FeatureStore:
    """
    在线特征存储引擎。

    接收一个 Redis 异步客户端实例，提供用户/物品特征的
    单点查询与批量查询能力。
    """

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    # ───────────────────────── 用户特征 ─────────────────────────────────

    async def get_user_features(self, user_id: int) -> Dict[str, Any]:
        """
        获取单用户的画像特征。

        优先使用 Redis HGETALL（哈希结构），若键不存在
        则回退到 GET JSON 字符串，最后兜底返回默认值。

        Parameters
        ----------
        user_id : int
            用户唯一标识。

        Returns
        -------
        Dict[str, Any]
            用户特征字典。
        """
        key = USER_FEATURE_KEY.format(user_id=user_id)

        # 尝试哈希结构
        data = await self.redis.hgetall(key)
        if data:
            return self._deserialize_user(data)

        # 尝试 JSON 字符串
        raw = await self.redis.get(key)
        if raw is not None:
            try:
                parsed = json.loads(raw)
                return {
                    **_DEFAULT_USER_FEATURES,
                    **parsed,
                    "user_id": user_id,
                }
            except json.JSONDecodeError:
                logger.warning(
                    "用户 [%s] 特征 JSON 解析失败，返回默认值。", user_id
                )

        logger.info("用户 [%s] 特征 Cache Miss，返回默认值。", user_id)
        return {**_DEFAULT_USER_FEATURES, "user_id": user_id}

    # ───────────────────────── 物品特征（批量） ──────────────────────────

    """
    工业级推荐请求的组 Batch 特性决定了我们要拉取大量候选物品特征。
    通过 Redis MGET / Pipeline，我们将原本 N 次网络 RTT（Round Trip Time）
    压缩到了 1 次，这是保障推荐链路 P99 延迟 < 50ms 的底层工程基石。
    """

    async def get_item_features_batch(
        self, item_ids: list[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        批量获取多个物品的特征。

        使用 Redis MGET 一次性拉取所有候选物品的 JSON 序列化特征，
        杜绝 for 循环逐条 GET 带来的 N 次 RTT 开销。

        Parameters
        ----------
        item_ids : list[int]
            候选物品 ID 列表。

        Returns
        -------
        Dict[int, Dict[str, Any]]
            {item_id: feature_dict} 的映射字典。
        """
        if not item_ids:
            return {}

        keys = [ITEM_FEATURE_KEY.format(item_id=item_id) for item_id in item_ids]

        # ── 一次网络 RTT 拉回所有物品特征 ──
        raw_values: list[Optional[str]] = await self.redis.mget(keys)

        result: Dict[int, Dict[str, Any]] = {}
        for item_id, raw in zip(item_ids, raw_values):
            if raw is not None:
                try:
                    parsed = json.loads(raw)
                    result[item_id] = {
                        **_DEFAULT_ITEM_FEATURES,
                        **parsed,
                        "item_id": item_id,
                    }
                except json.JSONDecodeError:
                    logger.warning(
                        "物品 [%s] 特征 JSON 解析失败，使用默认值。", item_id
                    )
                    result[item_id] = {**_DEFAULT_ITEM_FEATURES, "item_id": item_id}
            else:
                logger.info(
                    "物品 [%s] 特征 Cache Miss，返回默认值。", item_id
                )
                result[item_id] = {**_DEFAULT_ITEM_FEATURES, "item_id": item_id}

        return result

    # ───────────────────────── 辅助方法 ─────────────────────────────────

    @staticmethod
    def _deserialize_user(data: dict) -> Dict[str, Any]:
        """将 Redis HGETALL 返回的 bytes 字典转为 Python dict。"""
        return {
            "user_id": int(data.get("user_id", 0)),
            "age": int(data.get("age", 25)),
            "gender": data.get("gender", "unknown").decode()
            if isinstance(data.get("gender"), bytes)
            else data.get("gender", "unknown"),
            "city": data.get("city", "unknown").decode()
            if isinstance(data.get("city"), bytes)
            else data.get("city", "unknown"),
            "active_level": float(data.get("active_level", 1.0)),
        }

    @staticmethod
    def _deserialize_item(data: dict) -> Dict[str, Any]:
        """将 Redis HGETALL 返回的 bytes 字典转为 Python dict。"""
        return {
            "item_id": int(data.get("item_id", 0)),
            "category": data.get("category", "unknown").decode()
            if isinstance(data.get("category"), bytes)
            else data.get("category", "unknown"),
            "sub_category": data.get("sub_category", "unknown").decode()
            if isinstance(data.get("sub_category"), bytes)
            else data.get("sub_category", "unknown"),
            "price": float(data.get("price", 0.0)),
            "title": data.get("title", "").decode()
            if isinstance(data.get("title"), bytes)
            else data.get("title", ""),
        }


# ───────────────────── 模拟数据预热 ─────────────────────────────────────

_MOCK_USER_FEATURES: Dict[str, Any] = {
    "user_id": 1,
    "age": 28,
    "gender": "male",
    "city": "Beijing",
    "active_level": 4.7,
}

_MOCK_ITEM_CATEGORIES = [
    ("electronics", "smartphone"),
    ("electronics", "laptop"),
    ("clothing", "shoes"),
    ("clothing", "outerwear"),
    ("home", "kitchen"),
    ("home", "furniture"),
    ("books", "technology"),
    ("books", "fiction"),
    ("sports", "fitness"),
    ("sports", "outdoor"),
]


async def preload_mock_data(redis: Redis) -> None:
    """
    向 Redis 写入模拟特征数据，方便全链路集成测试。

    写入内容：
        - 用户 1 的画像特征（JSON 字符串）
        - 物品 101~150 的画像特征（JSON 字符串）

    Parameters
    ----------
    redis : Redis
        异步 Redis 客户端实例。
    """
    logger.info("开始预热 Mock 特征数据至 Redis ...")

    # ── 用户特征 ──
    user_key = USER_FEATURE_KEY.format(user_id=1)
    await redis.set(user_key, json.dumps(_MOCK_USER_FEATURES))
    logger.info("  ✅ 用户特征已写入: %s", user_key)

    # ── 物品特征 ──
    count = 0
    for item_id in range(101, 151):
        cat, sub_cat = _MOCK_ITEM_CATEGORIES[
            (item_id - 101) % len(_MOCK_ITEM_CATEGORIES)
        ]
        mock_item: Dict[str, Any] = {
            "item_id": item_id,
            "category": cat,
            "sub_category": sub_cat,
            "price": round(19.99 + (item_id - 101) * 3.5, 2),
            "title": f"Mock Item {item_id}",
        }
        item_key = ITEM_FEATURE_KEY.format(item_id=item_id)
        await redis.set(item_key, json.dumps(mock_item))
        count += 1

    logger.info("  ✅ 物品特征已写入 %d 条 (101~150)", count)
    logger.info("Mock 数据预热完成！")
