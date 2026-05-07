"""
LLM-RecFusion — Locust 全链路并发压测脚本

模拟线上高并发流量，持续轰炸 FastAPI 推荐接口 /api/v1/recommend。
压测数据可用于分析 P99 耗时、吞吐量 (RPS) 及系统瓶颈。

启动方式：
    locust -f load_test/locustfile.py

依赖：
    pip install locust
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


class RecommendationUser(HttpUser):
    """
    模拟推荐接口的真实用户请求。

    每个 Locust 用户以 0.1~0.5 秒的思考间隔持续向
    /api/v1/recommend 发送 POST 请求，构造符合
    RecommendRequest 规范的随机 Payload。
    """

    wait_time = between(0.1, 0.5)

    @task
    def recommend(self) -> None:
        """
        单次推荐请求任务。

        构造随机 user_id (1~10000) 与固定 top_k=10，
        发送 POST 请求并做轻量断言校验，避免 Locust 客户端
        自身成为压测瓶颈。
        """
        payload = {
            "user_id": random.randint(1, 10000),
            "top_k": 10,
        }

        with self.client.post(
            "/api/v1/recommend",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(
                    f"HTTP {resp.status_code} — 非 200 响应"
                )
                return

            try:
                body = resp.json()
            except ValueError:
                resp.failure("响应体非合法 JSON")
                return

            if "data" not in body:
                resp.failure("响应体缺少 data 字段")
                return
