"""
LLM-RecFusion — FastAPI 异步微服务入口

提供基于 FastAPI 的异步 Web 接口，包括：
- GET  /health             : 健康检查（K8s 存活探针）
- POST /api/v1/recommend   : 核心推荐接口（当前为 Mock 逻辑）
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI

from app.db import close_redis_pool, get_redis, init_redis_pool
from app.feature_store import FeatureStore, preload_mock_data
from app.recommender import RecommendEngine
from app.schemas import RecommendRequest, RecommendResponse

logger = logging.getLogger(__name__)

# ──────────────────────────── 应用实例与全局引擎 ────────────────────────

app = FastAPI(
    title="LLM-RecFusion Engine",
    version="1.0.0",
)

engine: RecommendEngine | None = None


# ──────────────────────────── 生命周期管理 ─────────────────────────────


@app.on_event("startup")
async def startup() -> None:
    """服务启动时初始化 Redis 连接池、推荐引擎并预热 Mock 特征数据。"""
    global engine
    logger.info("🚀 LLM-RecFusion 引擎启动，正在初始化基础设施 ...")

    # 1. 初始化 Redis 连接池
    await init_redis_pool()
    logger.info("  ✅ Redis 连接池已就绪")

    # 2. 初始化全局推荐引擎实例
    engine = RecommendEngine()
    logger.info("  ✅ 推荐引擎实例已创建")

    # 3. 预热 Mock 数据到 Redis（方便全链路测试）
    redis: aioredis.Redis = await anext(get_redis())
    await preload_mock_data(redis)
    await redis.aclose()

    logger.info("🎉 LLM-RecFusion 引擎启动完成！")


@app.on_event("shutdown")
async def shutdown() -> None:
    """服务关闭时安全释放 Redis 连接池。"""
    logger.info("🛑 LLM-RecFusion 引擎关闭，正在释放资源 ...")
    await close_redis_pool()
    logger.info("  ✅ Redis 连接池已释放")


# ──────────────────────────── 接口定义 ───────────────────────────────


@app.get("/health")
async def health_check(
    redis: aioredis.Redis = Depends(get_redis),
) -> dict:
    """
    K8s 存活探针 / 容器健康检查。

    通过 Depends 注入 Redis 实例，顺便验证 Redis 连通性。

    Parameters
    ----------
    redis : Redis
        由依赖注入系统提供的 Redis 客户端实例。
    """
    try:
        await redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok",
        "redis": "connected" if redis_ok else "disconnected",
    }


@app.post("/api/v1/recommend", response_model=RecommendResponse)
async def recommend(
    request: RecommendRequest,
    redis: aioredis.Redis = Depends(get_redis),
) -> RecommendResponse:
    """
    核心推荐接口。

    通过 FastAPI 的异步机制与 Pydantic 的数据防呆校验，我们将脏数据
    彻底拦截在算法链路之外。这确保了底层 C++ 算子和 PyTorch 引擎
    不会因为非法输入而发生 Core Dump。

    全链路流水线（由 RecommendEngine 编排）：
        Recall (100) → FeatureFetch → CoarseRank (100→20)
        → FineRank (20→20) → LLMRerank (20→top_k) → Response

    Parameters
    ----------
    request : RecommendRequest
        包含 user_id、top_k 及可选的 context_features。
    redis : Redis
        由依赖注入系统提供的 Redis 客户端实例。

    Returns
    -------
    RecommendResponse
        统一响应体，内含按得分降序排列的推荐列表。
    """
    global engine
    if engine is None:
        return RecommendResponse(
            code=500,
            message="RecommendEngine 尚未初始化",
            data=[],
        )

    store = FeatureStore(redis)
    items = await engine.get_recommendation(
        user_id=request.user_id,
        top_k=request.top_k,
        feature_store=store,
    )

    return RecommendResponse(code=200, message="success", data=items)
