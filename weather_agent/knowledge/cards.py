#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自编知识卡片：ENSO 领域高频问题的短答案。

每条都带来源链接。内容按公开资料整理，不是逐字转载；
数字尽量用阈值和年份，避免出现过期就错的观测值（观测值走工具查）。
"""

from __future__ import annotations

from typing import Any

CPC = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/"
CLIMATE = "https://www.climate.gov/enso"
NCEI = "https://www.ncei.noaa.gov/access/monitoring/enso/"
NCC = "https://www.ncc-cma.net/"

CARDS: list[dict[str, Any]] = [
    {
        "id": "def-oni",
        "title": "ONI 的定义",
        "text": "ONI（Oceanic Nino Index，海洋尼诺指数）是 Nino3.4 区（5S-5N，170W-120W）海表温度距平的三个月滑动平均。"
                "基准气候期用 1991-2020。它是 NOAA 判定 ENSO 状态的主用指标。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "def-season-label",
        "title": "季节标签的含义",
        "text": "ONI 的季节标签（DJF、JFM、到 NDJ）代表三个月窗口，三个字母分别是首月、次月、末月的缩写，"
                "所以 JAS 指 7 到 9 月，中心是 8 月。回答时不能把 JAS 说成 8 月的单月值。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "rule-5-season",
        "title": "五季判据",
        "text": "NOAA 的判定规则：ONI 连续五个重叠的三个月季节达到或超过正 0.5 摄氏度记 El Nino，"
                "连续五季达到或低于负 0.5 摄氏度记 La Nina。不足五季只能叫中性，或者用偏暖、偏冷来描述。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "def-nino-regions",
        "title": "Nino 分区的划分",
        "text": "Nino1+2 在东太平洋最东侧，Nino3 在中东太平洋，Nino4 在中西太平洋，"
                "Nino3.4 介于 3 与 4 之间，是 ONI 用的窗口。分型看的是 Nino3 与 Nino4 的相对大小。",
        "source_url": NCEI,
        "source_name": "NOAA NCEI",
    },
    {
        "id": "type-ep-cp",
        "title": "东部型与中部型",
        "text": "按暖中心位置分两型：东部型暖中心偏东太平洋，Nino3 距平明显大于 Nino4；"
                "中部型也叫 Modoki，暖中心偏中太平洋，Nino4 距平不低于 Nino3。"
                "两型的天气影响不同，所以只报 ONI 一个数不够。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "def-soi",
        "title": "SOI 的定义",
        "text": "SOI 即南方涛动指数，是塔希提与达尔文两地海平面气压差的标准化值。"
                "持续为负对应 El Nino，因为沃克环流减弱；持续为正对应 La Nina。"
                "它反映大气侧响应，可以与海洋侧的 ONI 交叉验证。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "mech-walker",
        "title": "沃克环流",
        "text": "沃克环流是赤道太平洋上空的东西向环流：西太平洋上升、东太平洋下沉，由海温东西差异驱动。"
                "El Nino 时东太平洋增暖、东西温差减小，环流减弱甚至反向，信风随之减弱。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "mech-bjerknes",
        "title": "Bjerknes 正反馈",
        "text": "海温升高让信风减弱，信风减弱又让暖水向东堆积，东太平洋进一步增暖。"
                "这个自我强化的循环叫 Bjerknes 正反馈，是 El Nino 能持续发展的核心机制；反向过程推动 La Nina。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "mech-onset",
        "title": "事件的触发与终结",
        "text": "触发常与西风爆发、季节内振荡东传有关；终结与赤道罗斯贝波、"
                "暖水东移后反射的开尔文波有关。这些是动力过程，单看指数看不出来。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "impact-easm",
        "title": "对东亚夏季风的影响",
        "text": "El Nino 发展年夏季，西太平洋副热带高压往往偏强偏西，东亚夏季风偏弱，"
                "中国长江流域及以南降水易偏多，华北易偏少。但这是统计倾向，个别年份会反。",
        "source_url": NCC,
        "source_name": "国家气候中心",
    },
    {
        "id": "impact-winter",
        "title": "对东亚冬季的影响",
        "text": "El Nino 年冬季，东亚冬季风通常偏弱、冷空气活动偏少，我国北方偏暖概率增大；"
                "La Nina 年则相反，寒潮更活跃。这类结论是概率性的，不能当确定性预报。",
        "source_url": NCC,
        "source_name": "国家气候中心",
    },
    {
        "id": "impact-typhoon",
        "title": "对西北太平洋台风的影响",
        "text": "El Nino 年西北太平洋台风生成位置通常偏东，生命史更长、强度偏强，登陆我国的数量往往偏少；"
                "La Nina 年生成位置偏西，登陆概率上升。",
        "source_url": NCC,
        "source_name": "国家气候中心",
    },
    {
        "id": "impact-global-temp",
        "title": "对全球温度的影响",
        "text": "El Nino 的增温效应有滞后，通常在事件次年对全球平均温度贡献最大；"
                "这就是厄尔尼诺次年往往更热这种说法的来源。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "def-la-nina",
        "title": "拉尼娜",
        "text": "La Nina 是 ENSO 的冷位相：Nino3.4 距平连续五季达到或低于负 0.5 摄氏度。"
                "中东太平洋信风增强、冷水上翻加强，通常伴随西太平洋降水偏多。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "def-neutral",
        "title": "中性状态",
        "text": "ONI 在负 0.5 到正 0.5 摄氏度之间，或者虽超阈值但不足五季，都归为中性。"
                "业务上会写中性偏暖或中性偏冷，但不能直接叫厄尔尼诺或拉尼娜。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "strength-bands",
        "title": "强度分级",
        "text": "常用分级按 ONI 峰值：0.5 到 0.9 弱，1.0 到 1.4 中等，1.5 到 1.9 强，2.0 及以上很强。"
                "不同机构的分档略有差异，引用时要说明用的是哪一套。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "history-strong",
        "title": "历史强事件",
        "text": "1950 年以来公认的强 El Nino 包括 1982/83、1997/98、2015/16；"
                "强 La Nina 包括 1973/74、1988/89、2010/11。给具体峰值要查序列，不要凭记忆。",
        "source_url": NCEI,
        "source_name": "NOAA NCEI",
    },
    {
        "id": "data-lag",
        "title": "数据为什么滞后",
        "text": "ONI 是三个月滑动平均，最新一个季节必须等窗口结束才能算；再加上观测资料处理时间，"
                "业务产品通常滞后一到一个半月。所以最新不等于现在。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "agency-diff",
        "title": "各机构判据不一致",
        "text": "NOAA 用 Nino3.4 三个月滑动平均和五季判据；日本气象厅用 Nino3 的五月滑动平均；"
                "澳大利亚气象局更强调 SOI 与信风。窗口、区域、阈值不同，结论可能不一致，这不是谁错了。",
        "source_url": CPC,
        "source_name": "NOAA CPC",
    },
    {
        "id": "term-modoki",
        "title": "Modoki 的含义",
        "text": "Modoki 是日语里相似但不同的意思，指中部型 El Nino：暖中心在中太平洋而不是东太平洋，"
                "对降水的影响型态与东部型明显不同。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "mech-coupling",
        "title": "海气耦合是否建立",
        "text": "判断事件是否成熟，除了看 ONI，还要看大气是否响应：SOI 是否持续转负、信风是否减弱、"
                "对流是否东移。只有海洋信号而大气未响应属于弱耦合，事件可能发展不起来。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "impact-iod",
        "title": "与印度洋偶极子的关联",
        "text": "ENSO 与印度洋偶极子存在统计关联：El Nino 年正 IOD 概率偏高，"
                "两者同相时会放大对周边降水的影响。但关联不等于因果，需要分别看。",
        "source_url": NCC,
        "source_name": "国家气候中心",
    },
    {
        "id": "climate-change",
        "title": "气候变化下的不确定性",
        "text": "观测上 El Nino 的强度与频率是否有长期趋势，目前结论不一致；"
                "多数模式预估强 El Nino 的频率上升，但信噪比低。回答这类问题要说明研究存在分歧。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
    {
        "id": "forecast-limit",
        "title": "预报的有效期限",
        "text": "ENSO 是短期气候预测里可预报性较高的现象，动力模式在越过春季可预报性障碍后，"
                "能给出几个月的有效预报；超过一年基本没有技巧。任何明年必是厄尔尼诺的说法都不可信。",
        "source_url": CLIMATE,
        "source_name": "NOAA Climate.gov",
    },
]
