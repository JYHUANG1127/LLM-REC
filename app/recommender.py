"""
LLM-RecFusion — Stage 6: 全链路异步推荐引擎

将多路召回、Redis 特征拉取、粗排、精排和 vLLM 大模型重排
串联成一个 End-to-End 的异步流水线。

架构总览:
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Recall   │→  │ Feature  │→  │ Coarse   │→  │ Fine     │→  │ LLM      │
│ (100)    │   │ Fetch    │   │ Rank     │   │ Rank     │   │ Rerank   │
│          │   │ (100)    │   │(100→20)  │   │(20→20)   │   │(20→10)   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘

每阶段超参数可通过类属性灵活调整，支持一键切换 Mock / 真实模型。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import numpy as np
import torch

from app.feature_store import FeatureStore
from app.schemas import ItemScore

logger = logging.getLogger(__name__)

# ──────────────────── 全局随机状态（保证同一场景结果可复现）────────────────────
_RNG = np.random.default_rng(42)


class RecommendEngine:
    """
    全链路推荐引擎。

    编排多路召回 → 特征拉取 → 粗排截断 → 精排打分 → LLM 重排
    的异步流水线，当前各阶段均采用 Mock 模型模拟真实行为，
    后续可逐阶段替换为已训练的 PyTorch 模型权重。

    Parameters
    ----------
    recall_top_k : int
        召回阶段候选物品数量，默认 100。
    coarse_top_k : int
        粗排阶段截断数量，默认 20。
    embed_dim : int
        模拟 Embedding 维度，默认 16。
    """

    def __init__(
        self,
        recall_top_k: int = 100,
        coarse_top_k: int = 20,
        embed_dim: int = 16,
    ) -> None:
        self.recall_top_k: int = recall_top_k
        self.coarse_top_k: int = coarse_top_k
        self.embed_dim: int = embed_dim
        self.coarse_hidden: int = 64  # 粗排 MLP 隐藏层维度
        logger.info(
            "RecommendEngine 初始化完成 — "
            "recall_top_k=%d, coarse_top_k=%d, embed_dim=%d",
            self.recall_top_k,
            self.coarse_top_k,
            self.embed_dim,
        )

    # ════════════════════════════════════════════════════════════
    # 公开接口：全链路编排
    # ════════════════════════════════════════════════════════════

    async def get_recommendation(
        self,
        user_id: int,
        top_k: int,
        feature_store: FeatureStore,
    ) -> List[ItemScore]:
        """
        执行全链路推荐流水线，返回最终排序后的推荐结果。

        流水线包含以下步骤：
          1. 多路召回 (Recall)       — Mock 逻辑返回 100 个候选
          2. 特征并行拉取 (Feature)   — Redis MGET 批量拉取
          3. 粗排截断 (Coarse)       — Mock Wide&Deep, 100→20
          4. 精排打分 (Fine)         — Mock DIN+Dice, 20→20
          5. LLM 重排 (LLM Rerank)   — Mock Listwise, 20→top_k
          6. 响应组装 (Response)     — 封装为 List[ItemScore]

        Parameters
        ----------
        user_id : int
            请求推荐的目标用户 ID。
        top_k : int
            最终返回的推荐物品数量。
        feature_store : FeatureStore
            已注入 Redis 连接的异步特征存储引擎。

        Returns
        -------
        List[ItemScore]
            按得分降序排列的推荐结果列表。
        """
        # ─────────────────────────────────────────────────────
        # Step 1: 多路召回 (Recall)
        # ─────────────────────────────────────────────────────
        # 基于 user_id 的确定性采样，模拟多路召回融合后的 100 个候选。
        recall_items: List[int] = self._mock_recall(user_id)
        logger.info(
            "Step 1 ✅  召回阶段完成 — 候选物品数: %d",
            len(recall_items),
        )

        # ─────────────────────────────────────────────────────
        # Step 2: 特征并行拉取 (Feature Fetching)
        # ─────────────────────────────────────────────────────
        # 一次网络 RTT 拉回所有候选物品特征（Redis MGET）。
        item_features: Dict[int, Dict[str, Any]] = (
            await feature_store.get_item_features_batch(recall_items)
        )
        logger.info(
            "Step 2 ✅  特征拉取完成 — 拉取物品数: %d",
            len(item_features),
        )

        # ─────────────────────────────────────────────────────
        # Step 3: 粗排截断 (Coarse Ranking)
        # ─────────────────────────────────────────────────────
        # 特征字典 → Tensor → Mock Wide&Deep → Top-20。
        candidate_ids, scores = self._coarse_rank(
            item_features, recall_items,
        )
        coarse_top20: List[int] = candidate_ids[: self.coarse_top_k]
        logger.info(
            "Step 3 ✅  粗排完成 — %d→%d, 最高分: %.4f",
            len(candidate_ids),
            len(coarse_top20),
            scores[0] if len(scores) > 0 else 0.0,
        )

        # ─────────────────────────────────────────────────────
        # Step 4: 精排打分 (Fine Ranking)
        # ─────────────────────────────────────────────────────
        # 拼接用户特征 + 物品特征，Mock DIN + Dice 前向传播。
        user_feat: Dict[str, Any] = await feature_store.get_user_features(user_id)
        fine_scores: Dict[int, float] = self._fine_rank(
            coarse_top20, item_features, user_feat,
        )
        fine_ranked: List[int] = sorted(
            fine_scores, key=lambda x: fine_scores[x], reverse=True,
        )
        logger.info(
            "Step 4 ✅  精排完成 — 最高分: %.4f",
            fine_scores[fine_ranked[0]] if fine_ranked else 0.0,
        )

        # ─────────────────────────────────────────────────────
        # Step 5: vLLM 大模型重排 (LLM Reranking)
        # ─────────────────────────────────────────────────────
        # 精排 Top-20 → Listwise Prompt → 异步 LLM 调用 → Top-K。
        llm_ranked: List[int] = await self._llm_rerank(
            fine_ranked, item_features,
        )
        final_items: List[int] = llm_ranked[:top_k]
        logger.info(
            "Step 5 ✅  LLM 重排完成 — %d→%d",
            len(fine_ranked),
            len(final_items),
        )

        # ─────────────────────────────────────────────────────
        # Step 6: 组装响应
        # ─────────────────────────────────────────────────────
        result: List[ItemScore] = [
            ItemScore(
                item_id=item_id,
                score=round(fine_scores.get(item_id, 0.0), 6),
            )
            for item_id in final_items
        ]
        logger.info(
            "Step 6 ✅  响应组装完成 — 返回 %d 条推荐结果",
            len(result),
        )

        return result

    # ════════════════════════════════════════════════════════════
    # 子步骤实现
    # ════════════════════════════════════════════════════════════

    # ─────────────────────── Step 1 ───────────────────────────

    def _mock_recall(self, user_id: int) -> List[int]:
        """
        模拟多路召回输出。

        用 user_id 播种确定性随机数生成器，从 101~1400 的候选中
        无放回采样 100 个物品 ID，模拟多路召回融合后的结果。

        Parameters
        ----------
        user_id : int
            用户 ID，用作随机种子保证结果可复现。

        Returns
        -------
        List[int]
            召回阶段输出的候选物品 ID 列表。
        """
        rng = np.random.default_rng(seed=user_id)
        pool: np.ndarray = np.arange(101, 1401)
        chosen: np.ndarray = rng.choice(
            pool, size=self.recall_top_k, replace=False,
        )
        return sorted(chosen.tolist())

    # ─────────────────────── Step 3 ───────────────────────────

    def _coarse_rank(
        self,
        item_features: Dict[int, Dict[str, Any]],
        recall_items: List[int],
    ) -> tuple[List[int], np.ndarray]:
        """
        粗排阶段：模拟 Wide&Deep 前向传播。

        将每个物品的字典特征编码为稠密向量，分别通过 Wide（线性）
        和 Deep（两层 MLP + ReLU）通道计算得分，融合后取 Top-K。

        Parameters
        ----------
        item_features : Dict[int, Dict[str, Any]]
            {item_id: feature_dict} 映射。
        recall_items : List[int]
            召回阶段输出的候选物品 ID 列表。

        Returns
        -------
        tuple[List[int], np.ndarray]
            (按得分降序排列的物品 ID 列表, 对应得分数组)。
        """
        # --- 特征编码: dict → 固定维度 numpy 向量 ---
        feature_dim: int = self.embed_dim + 4  # 16 + 4 = 20 维
        feature_matrix: np.ndarray = np.zeros(
            (len(recall_items), feature_dim), dtype=np.float32,
        )

        for i, item_id in enumerate(recall_items):
            feat: Dict[str, Any] = item_features.get(item_id, {})
            # 数值特征归一化
            price_norm: float = min(feat.get("price", 0.0) / 500.0, 1.0)
            # 类别特征的确定性哈希映射
            cat_hash: float = (
                hash(feat.get("category", "")) % 1000
            ) / 1000.0
            sub_hash: float = (
                hash(feat.get("sub_category", "")) % 1000
            ) / 1000.0
            title_len: float = min(
                len(str(feat.get("title", ""))) / 100.0, 1.0,
            )

            # 手工拼接的 4 维稠密特征
            dense_feat: np.ndarray = np.array(
                [price_norm, cat_hash, sub_hash, title_len],
                dtype=np.float32,
            )

            # 模拟 Embedding 层（随机投影矩阵）
            embedding: np.ndarray = _RNG.normal(
                0, 0.1, size=self.embed_dim,
            ).astype(np.float32)

            feature_matrix[i] = np.concatenate([dense_feat, embedding])

        # --- Wide 部分: 线性投影 ---
        wide_weight: np.ndarray = _RNG.normal(
            0, 0.1, size=feature_dim,
        ).astype(np.float32)
        wide_bias: float = 0.1
        wide_score: np.ndarray = feature_matrix @ wide_weight + wide_bias  # [100]

        # --- Deep 部分: 两层 MLP + ReLU ---
        hidden_w: np.ndarray = _RNG.normal(
            0, 0.1, size=(feature_dim, self.coarse_hidden),
        ).astype(np.float32)
        hidden_b: np.ndarray = np.zeros(self.coarse_hidden, dtype=np.float32)
        output_w: np.ndarray = _RNG.normal(
            0, 0.1, size=self.coarse_hidden,
        ).astype(np.float32)

        hidden: np.ndarray = np.maximum(
            feature_matrix @ hidden_w + hidden_b, 0.0,  # ReLU
        )
        deep_score: np.ndarray = hidden @ output_w  # [100]

        # --- Wide + Deep 融合得分 ---
        combined_scores: np.ndarray = wide_score + deep_score

        # --- Top-K 选取 ---
        topk_indices: np.ndarray = np.argsort(combined_scores)[::-1]
        sorted_ids: List[int] = [
            recall_items[i] for i in topk_indices.tolist()
        ]
        sorted_scores: np.ndarray = combined_scores[topk_indices]

        return sorted_ids, sorted_scores

    # ─────────────────────── Step 4 ───────────────────────────

    def _fine_rank(
        self,
        coarse_top20: List[int],
        item_features: Dict[int, Dict[str, Any]],
        user_feat: Dict[str, Any],
    ) -> Dict[int, float]:
        """
        精排阶段：模拟 DIN + Dice 的前向传播。

        对每个粗排通过的候选物品，执行以下流程:
          1. 目标物品 Embedding + 用户特征 Embedding
          2. NoSoftmax Target Attention（模拟用户兴趣提取）
          3. 特征拼接 + MLP（带 Dice 激活）→ Logit

        Parameters
        ----------
        coarse_top20 : List[int]
            粗排截断后的 Top-20 物品 ID。
        item_features : Dict[int, Dict[str, Any]]
            {item_id: feature_dict} 映射。
        user_feat : Dict[str, Any]
            当前用户的画像特征字典。

        Returns
        -------
        Dict[int, float]
            {item_id: 精排得分} 映射。
        """
        device = torch.device("cpu")

        # --- 用户侧特征编码 ---
        age_norm: float = min(user_feat.get("age", 25) / 100.0, 1.0)
        active_norm: float = min(
            user_feat.get("active_level", 1.0) / 5.0, 1.0,
        )
        user_dense: torch.Tensor = torch.tensor(
            [age_norm, active_norm], dtype=torch.float32, device=device,
        )
        user_emb: torch.Tensor = torch.tensor(
            _RNG.normal(0, 0.1, size=self.embed_dim),
            dtype=torch.float32, device=device,
        )  # [D]

        # --- 模拟用户历史行为序列（SIM GSU 截断后的 20 个行为）---
        num_history: int = 20
        hist_emb: torch.Tensor = torch.tensor(
            _RNG.normal(0, 0.1, size=(num_history, self.embed_dim)),
            dtype=torch.float32, device=device,
        )  # [20, D]

        # --- 对每个候选物品计算精排得分 ---
        scores: Dict[int, float] = {}

        for item_id in coarse_top20:
            feat: Dict[str, Any] = item_features.get(item_id, {})
            price_norm: float = min(feat.get("price", 0.0) / 500.0, 1.0)

            # 目标物品 Embedding
            item_emb: torch.Tensor = torch.tensor(
                _RNG.normal(0, 0.1, size=self.embed_dim),
                dtype=torch.float32, device=device,
            )  # [D]

            # ══════════════════════════════════════════════════
            # 模拟 NoSoftmax Target Attention
            # ══════════════════════════════════════════════════
            # 架构: concat(q, k, q-k, q*k) → MLP → scores → Σ(s_i * k_i)
            query: torch.Tensor = item_emb.unsqueeze(0).unsqueeze(1)  # [1, 1, D]
            keys: torch.Tensor = hist_emb.unsqueeze(0)               # [1, 20, D]

            cat_feat: torch.Tensor = torch.cat(
                [
                    query.expand(-1, num_history, -1),
                    keys,
                    query.expand(-1, num_history, -1) - keys,
                    query.expand(-1, num_history, -1) * keys,
                ],
                dim=-1,
            )  # [1, 20, 4D]

            # Attention MLP: 4D → 80 → 40 → 1
            attn_w1: torch.Tensor = torch.tensor(
                _RNG.normal(0, 0.1, size=(4 * self.embed_dim, 80)),
                dtype=torch.float32, device=device,
            )
            attn_w2: torch.Tensor = torch.tensor(
                _RNG.normal(0, 0.1, size=(80, 40)),
                dtype=torch.float32, device=device,
            )
            attn_w3: torch.Tensor = torch.tensor(
                _RNG.normal(0, 0.1, size=(40, 1)),
                dtype=torch.float32, device=device,
            )

            attn_h: torch.Tensor = torch.relu(cat_feat @ attn_w1)   # [1, 20, 80]
            attn_h = torch.relu(attn_h @ attn_w2)                   # [1, 20, 40]
            attn_scores: torch.Tensor = attn_h @ attn_w3             # [1, 20, 1]

            # 加权求和（无 Softmax，保留绝对兴趣强度）
            interest: torch.Tensor = (
                (attn_scores * keys).sum(dim=1)  # [1, D]
            )

            # ══════════════════════════════════════════════════
            # 特征拼接 + MLP（带 Dice 激活）→ Logit
            # ══════════════════════════════════════════════════
            # Concat[item_emb, user_emb, interest] → MLP → Dice → Logit
            combined: torch.Tensor = torch.cat(
                [
                    item_emb.unsqueeze(0),  # [1, D]
                    user_emb.unsqueeze(0),   # [1, D]
                    interest,                # [1, D]
                ],
                dim=-1,
            )  # [1, 3D]

            # MLP: 3D → 200(ReLU) → 80(Dice) → 1(Logit)
            mlp_w1: torch.Tensor = torch.tensor(
                _RNG.normal(0, 0.1, size=(3 * self.embed_dim, 200)),
                dtype=torch.float32, device=device,
            )
            mlp_w2: torch.Tensor = torch.tensor(
                _RNG.normal(0, 0.1, size=(200, 80)),
                dtype=torch.float32, device=device,
            )
            mlp_w3: torch.Tensor = torch.tensor(
                _RNG.normal(0, 0.1, size=(80, 1)),
                dtype=torch.float32, device=device,
            )

            h1: torch.Tensor = torch.relu(combined @ mlp_w1)  # [1, 200]
            h1_dice: torch.Tensor = self._mock_dice(h1)       # [1, 200]
            h2: torch.Tensor = torch.relu(h1_dice @ mlp_w2)   # [1, 80]
            h2_dice: torch.Tensor = self._mock_dice(h2)       # [1, 80]
            logit: torch.Tensor = h2_dice @ mlp_w3             # [1, 1]

            # 加入价格先验偏置
            score: float = logit.item() + price_norm * 0.5
            scores[item_id] = score

        return scores

    @staticmethod
    def _mock_dice(x: torch.Tensor) -> torch.Tensor:
        """
        模拟 Dice (Data-adaptive Activation) 激活函数。

        Dice 核心公式:
            norm_x = (x - mean) / sqrt(var + epsilon)
            p      = sigmoid(norm_x)
            f(x)   = p * x + (1 - p) * alpha * x

        这里 alpha 固定为 0.0（初始状态），体现 Dice 初期的"几乎全量通过"特性。
        """
        if x.numel() <= 1:
            return x
        mean: torch.Tensor = x.mean(dim=0, keepdim=True)
        var: torch.Tensor = x.var(dim=0, keepdim=True, unbiased=False)
        norm_x: torch.Tensor = (x - mean) / torch.sqrt(var + 1e-8)
        p: torch.Tensor = torch.sigmoid(norm_x)
        alpha: float = 0.0  # 初始化为 0，信号几乎全量通过
        return p * x + (1.0 - p) * alpha * x

    # ─────────────────────── Step 5 ───────────────────────────

    async def _llm_rerank(
        self,
        fine_ranked: List[int],
        item_features: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        """
        vLLM 大模型 Listwise 重排。

        将精排 Top-20 转换为结构化文本 Prompt，
        异步调用大模型进行多样性感知的重新排序。

        在实际部署中，此处应替换为真实的 vLLM HTTP 调用:
            POST http://vllm-server:8000/v1/completions
            {
                "model": "qwen2.5-7b-instruct",
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.1
            }

        Parameters
        ----------
        fine_ranked : List[int]
            精排阶段输出的 Top-20 物品 ID（按得分降序）。
        item_features : Dict[int, Dict[str, Any]]
            {item_id: feature_dict} 映射，用于构建 Prompt。

        Returns
        -------
        List[int]
            LLM 重排后的物品 ID 列表。
        """
        # --- 构建 Listwise Prompt ---
        lines: List[str] = []
        for i, item_id in enumerate(fine_ranked):
            feat: Dict[str, Any] = item_features.get(item_id, {})
            lines.append(
                f"[{i+1}] Item {item_id} — "
                f"Category: {feat.get('category', 'unknown')}, "
                f"Price: ${feat.get('price', 0.0):.2f}, "
                f"Title: {feat.get('title', 'N/A')}"
            )

        prompt: str = (
            "You are a recommendation reranking assistant.\n"
            "Below is a list of candidate items for the user.\n"
            "Please re-rank them based on diversity and relevance.\n"
            f"Input items:\n{chr(10).join(lines)}\n"
            "Output the re-ranked item IDs in descending order of preference, "
            "comma-separated."
        )
        logger.debug("LLM Rerank Prompt:\n%s", prompt)

        # --- 异步 Mock vLLM 调用 ---
        # 模拟 ~10ms 网络延迟（实际 vLLM 首 token 延迟约 30-50ms）
        await asyncio.sleep(0.01)

        # Mock 逻辑: 基于类目做贪心多样性重排（MMR 思想）
        reranked: List[int] = self._diversity_rerank(fine_ranked, item_features)
        logger.info(
            "  ↪ LLM 重排完成 — 输入 %d 项, 输出 %d 项",
            len(fine_ranked),
            len(reranked),
        )

        return reranked

    @staticmethod
    def _diversity_rerank(
        items: List[int],
        item_features: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        """
        基于类目的贪心多样性重排。

        采用最大边际相关性 (MMR) 的简化版：按类目分组后轮询各分组，
        使推荐结果在精排得分之外兼顾类目多样性，避免同类物品连续扎堆。

        Parameters
        ----------
        items : List[int]
            按精排得分降序排列的物品 ID 列表。
        item_features : Dict[int, Dict[str, Any]]
            {item_id: feature_dict} 映射。

        Returns
        -------
        List[int]
            多样性重排后的物品 ID 列表。
        """
        # 按类目分组
        cat_groups: Dict[str, List[int]] = {}
        for item_id in items:
            cat: str = item_features.get(item_id, {}).get(
                "category", "unknown",
            )
            cat_groups.setdefault(cat, []).append(item_id)

        # 将各组转为列表，按组大小降序排列
        iterators: List[List[int]] = list(cat_groups.values())
        iterators.sort(key=lambda x: len(x), reverse=True)

        # 轮询各分组，最大化交错多样性
        result: List[int] = []
        max_len: int = max(len(g) for g in iterators)
        for i in range(max_len):
            for group in iterators:
                if i < len(group):
                    result.append(group[i])

        # 安全兜底：补回任何遗漏的物品
        remaining: List[int] = [x for x in items if x not in result]
        result.extend(remaining)

        return result
