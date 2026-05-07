"""
LLM-RecFusion — Redis 连接池管理

为全链路提供异步 Redis 连接池，避免每次请求都创建新连接。
通过 ConnectionPool 复用 TCP 连接，是保障推荐服务高吞吐、
低延迟的底层基础设施。
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool, Redis

logger = logging.getLogger(__name__)

# ───────────────────── 可配置常量（可通过环境变量覆盖）─────────────────────
REDIS_URL: str = "redis://localhost:6379/0"

# ───────────────────── 连接池（模块级单例）───────────────────────────────
_pool: ConnectionPool | None = None


async def init_redis_pool(redis_url: str | None = None) -> None:
    """
    初始化 Redis 连接池。

    参数允许通过环境变量或配置中心传入 redis_url，
    默认指向本地 6379 端口的 db 0。

    Parameters
    ----------
    redis_url : str, optional
        Redis 连接字符串，例如 "redis://:password@host:port/db"。
    """
    global _pool
    url = redis_url or REDIS_URL
    _pool = ConnectionPool.from_url(url, decode_responses=True)
    # 快速验证连接是否可用
    r = Redis(connection_pool=_pool)
    await r.ping()
    await r.aclose()
    logger.info("Redis 连接池初始化成功 — %s", url)


async def close_redis_pool() -> None:
    """安全释放 Redis 连接池，防止资源泄漏。"""
    global _pool
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
        logger.info("Redis 连接池已释放。")


async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    FastAPI Depends 注入依赖。

    Yields
    ------
    Redis
        一个从共享连接池借出的 Redis 异步客户端实例。
        请求结束后自动归还到连接池。
    """
    if _pool is None:
        raise RuntimeError(
            "Redis 连接池尚未初始化！请先调用 init_redis_pool()。"
        )
    r = Redis(connection_pool=_pool)
    try:
        yield r
    finally:
        await r.aclose()
