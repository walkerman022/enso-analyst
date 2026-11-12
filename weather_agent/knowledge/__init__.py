#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""知识库：把"ENSO 领域知识"变成可检索、可引用的片段。

三层语料：
1. 官方公开文档（NOAA Climate.gov / CPC / NCEI、国家气候中心）——构建时抓取，只存摘要片段 + 原文链接
2. 百科类（Wikipedia ENSO 条目）——CC BY-SA，保留署名与链接
3. 自编卡片（按公开资料整理）——覆盖定义、判据、机制、影响等高频问题

为什么不用向量库：
语料只有几十到几百个片段，中文按二元组做词频检索已经够准，
上 embedding 要多一个模型依赖、多一层延迟，收益不成正比。

为什么每个片段都带来源：
知识类回答最大的风险不是"不知道"，是"编一段听起来对的话"。
片段带 id、来源、抓取时间，模型引用时必须写 [kb-xx]，护栏再校验这个 id 是否真的被召回过。
"""

from __future__ import annotations

from weather_agent.knowledge.store import (
    add_chunks,
    count_chunks,
    get_chunks,
    init_db,
    search,
)
from weather_agent.knowledge.cards import CARDS

__all__ = ["CARDS", "add_chunks", "count_chunks", "get_chunks", "init_db", "search"]
