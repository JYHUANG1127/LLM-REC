"""
LLM-RecFusion — Pydantic Schemas

强类型数据校验层，将脏数据彻底拦截在算法链路之外。
所有请求 / 响应模型均附带清晰的 Field description，
用于自动生成大厂级别的 Swagger / OpenAPI 文档。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ────────────────────────────── Request ──────────────────────────────


class RecommendRequest(BaseModel):
    """推荐请求体"""

    user_id: int = Field(
        ...,
        gt=0,
        description="用户唯一标识，必须大于 0。",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="返回候选物品的数量，取值范围 [1, 50]，默认 10。",
    )
    context_features: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "用户当前实时上下文特征，例如："
            '{"network_latency_ms": 120.5, "geo_latitude": 39.90, "geo_longitude": 116.40}。'
        ),
    )


# ────────────────────────────── Response ─────────────────────────────


class ItemScore(BaseModel):
    """单条推荐物品及其得分"""

    item_id: int = Field(
        ...,
        description="推荐物品的唯一标识。",
    )
    score: float = Field(
        ...,
        description="模型对该物品的预估得分 / 置信度。",
    )


class RecommendResponse(BaseModel):
    """推荐接口的统一响应体"""

    code: int = Field(
        default=200,
        description="业务状态码，200 表示成功。",
    )
    message: str = Field(
        default="success",
        description="业务提示信息。",
    )
    data: List[ItemScore] = Field(
        ...,
        description="推荐结果列表，按得分降序排列。",
    )
