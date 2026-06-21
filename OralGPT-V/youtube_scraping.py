#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube智能视频下载器
参考B站脚本，支持AI关键词生成、视频相关性判断与批量下载
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Any
from urllib.parse import urlparse, quote_plus
import http.client
import random

from bs4 import BeautifulSoup
try:
    from selenium import webdriver
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
except ModuleNotFoundError:
    webdriver = None
    WebDriverException = RuntimeError
    Options = None
    Service = None

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

REPO_ROOT = Path(__file__).resolve().parent
TRACKING_ROOT = Path(os.environ.get("TRACKING_ROOT", REPO_ROOT / "tracking"))
VIDEO_YOUTUBE_ROOT = Path(os.environ.get("VIDEO_YOUTUBE_ROOT", REPO_ROOT / "youtube_videos"))


VIDEO_FILE_EXTENSIONS = {".mp4", ".mkv", ".webm", ".flv", ".mov", ".m4v"}


CATEGORY_CONFIGS = {
    "cinematic_arts": {
        "generate_keywords_system_prompt": """你是一个YouTube搜索专家。根据用户主题，生成高质量的搜索关键词。

关键词要求：
1. 聚焦于电影拉片、电影解说、视听语言、导演镜头技法
2. 包含电影技法术语，如：蒙太奇、镜头语言、场面调度、长镜头、希区柯克、色彩构图等
3. 贴合YouTube用户真实搜索习惯，兼顾中英组合（如："film breakdown 拉片"）
4. 多样化，覆盖不同电影流派与导演风格

输出JSON:
{
    "keywords": ["关键词1", "关键词2", ...]
}""",
        "generate_keywords_user_prompt": """主题: {topic}

    请生成{num_keywords}个最有效的YouTube搜索关键词。尽量覆盖拉片、解说和电影技法。
    语言要求: {lang_rule}

    必备关注点（可自由组合）：
    - 摄影（Cinematography）：镜头景别（Long Shot/Extreme Close-up）、机位角度（Dutch Angle/Bird's Eye）、运动方式（Dolly Zoom/Tracking）、光影色彩（High Key/Low Key，色彩心理）、构图法则（Rule of Thirds/Leading Lines）。
    - 剪辑（Editing）：蒙太奇、长镜头、跳切、平行剪辑、闪回、转场（Match Cut/Smash Cut）、库里肖夫效应。
    - 电影场面调度（Mise-en-scène）：场景与道具符号、人物走位与调度、服化道呈现人物关系。
    - 声音（Sound Design）：声画关系（同步/反差）、叙事来源（Diegetic vs Non-diegetic）、声音桥、环境声、利用静默。
    - 可结合导演或大师案例解释这些技法。
    """,
        "judge_relevance_system_prompt": """你是资深电影教育顾问。请判断给定的YouTube视频是否满足用户的电影拉片/技法学习需求。

输出JSON:
{
    "is_relevant": true/false,
    "reason": "原因",
    "confidence": 0.0-1.0
}""",
        "judge_relevance_user_hint": "请判断该视频是否聚焦电影拉片、解说或电影技法，并输出理由。",
        "variant_focus": "电影摄影、剪辑、场面调度和声音设计",
        "fallback_map": {
            "en": [
                "cinematography shot size breakdown",
                "dutch angle vs bird's eye analysis",
                "dolly zoom vertigo effect tutorial",
                "high key low key lighting study",
                "rule of thirds leading lines composition",
                "montage cross cutting film editing",
                "kuleshov effect explanation",
                "mise en scene blocking symbolism",
                "diegetic vs non diegetic sound design",
                "sound bridge ambient noise film",
            ],
            "zh-CN": [
                "电影摄影 镜头景别 解析",
                "荷兰式角度 鸟瞰镜头 分析",
                "推轨变焦 眩晕镜头 教程",
                "高调光 低调光 色彩心理",
                "三分法 引导线 构图",
                "蒙太奇 平行剪辑 技法",
                "库里肖夫 效应 讲解",
                "场面调度 走位 象征",
                "内景声 外景声 声音设计",
                "声音桥 环境声 静默 运用",
            ],
            "zh-TW": [
                "電影攝影 鏡頭景別 拆解",
                "荷蘭式角度 鳥瞰 構圖 解析",
                "推軌變焦 眩暈鏡頭 教學",
                "高調光 低調光 色彩心理",
                "黃金分割 引導線 架構",
                "蒙太奇 平行剪接 技法",
                "庫里肖夫 效應 說明",
                "場面調度 走位 象徵",
                "內景聲 外景聲 聲音設計",
                "聲音橋 氛圍聲 靜默",
            ],
        },
    },
    "static_visual_arts": {
        "generate_keywords_system_prompt": """你是一个专注于视觉艺术领域的视频搜索专家。根据用户主题，生成高质量的搜索关键词。

关键词要求：
1. 聚焦于绘画赏析、摄影评论、数字艺术分析、艺术史讲解类视频
2. 包含视觉艺术术语，如：构图、色彩理论、光影、透视、笔触、风格化、艺术流派、摄影语言等
3. 贴合平台用户真实搜索习惯
4. 多样化，覆盖传统绘画、摄影、数字艺术等不同媒介

输出JSON:
{
    "keywords": ["关键词1", "关键词2", ...]
}""",
        "generate_keywords_user_prompt": """主题: {topic}

请生成{num_keywords}个最有效的搜索关键词，聚焦静态视觉艺术的 video essay / 赏析 / 解说类内容。
语言要求: {lang_rule}

必备关注点（可自由组合）：
- 绘画技法：构图法则（黄金比例、三角构图）、色彩理论（互补色、色温）、光影处理（明暗对比 / Chiaroscuro）、透视（线性透视、大气透视）、笔触风格（印象派、点彩派）
- 艺术史与流派：文艺复兴、巴洛克、印象派、现代主义、超现实主义、当代数字艺术
- 摄影语言：景深（浅景深/深景深）、曝光三角（ISO/快门/光圈）、构图规则（Rule of Thirds/Leading Lines）、黑白摄影、纪实摄影 vs 艺术摄影
- 数字艺术：概念设计（Concept Art）、像素艺术、AI 生成艺术的批评与赏析
- 可结合具体艺术家或作品案例分析（如：伦勃朗光、维米尔的光影、安塞尔·亚当斯的区域曝光法）""",
        "judge_relevance_system_prompt": """你是一个视觉艺术视频内容分析专家。你的任务是判断给定视频是否属于静态视觉艺术（绘画、素描、数字艺术、摄影）的赏析、解说、评论或教学类 video essay。

判断标准：
- ✅ 符合：对绘画/摄影/数字艺术作品进行深度分析、技法讲解、艺术史背景解读、风格比较
- ❌ 不符合：纯粹的绘画教程（step-by-step）、修图工具教学（PS/LR 操作）、娱乐向 Vlog、购买设备评测

输出JSON:
{
    "is_relevant": true/false,
    "reason": "判断理由",
    "confidence": 0.0-1.0
}""",
        "judge_relevance_user_hint": "请判断该视频是否聚焦静态视觉艺术（绘画/摄影/数字艺术）的赏析或解说，并输出理由。",
        "variant_focus": "绘画赏析、摄影评论、数字艺术分析和艺术史",
        "fallback_map": {
            "en": [
                "painting composition analysis art essay",
                "color theory visual art breakdown",
                "chiaroscuro light shadow oil painting",
                "photography visual language analysis",
                "concept art design principles essay",
                "impressionism post-impressionism art history",
                "rule of thirds photography composition",
                "digital art critique aesthetic analysis",
                "fine art photography essay depth of field",
                "art history documentary renaissance baroque",
            ],
            "zh-CN": [
                "绘画构图赏析 视频解说",
                "色彩理论 互补色 色温 分析",
                "明暗对比 伦勃朗光 油画技法",
                "摄影语言 景深 构图 解析",
                "概念艺术 设计原理 视频评论",
                "印象派 后印象派 艺术史",
                "三分法 摄影构图 解说",
                "数字艺术 审美分析 视频论文",
                "艺术摄影 纪实摄影 比较赏析",
                "文艺复兴 巴洛克 艺术史讲解",
            ],
            "zh-TW": [
                "繪畫構圖賞析 視頻解說",
                "色彩理論 互補色 色溫 分析",
                "明暗對比 倫勃朗光 油畫技法",
                "攝影語言 景深 構圖 解析",
                "概念藝術 設計原理 視頻評論",
                "印象派 後印象派 藝術史",
                "三分法 攝影構圖 解說",
                "數位藝術 審美分析 視頻論文",
                "藝術攝影 紀實攝影 比較賞析",
                "文藝復興 巴洛克 藝術史講解",
            ],
        },
    },
    "stage_performing_arts": {
        "generate_keywords_system_prompt": """你是一个专注于舞台表演艺术领域的视频搜索专家。根据用户主题，生成高质量的搜索关键词。

关键词要求：
1. 聚焦于戏剧、歌剧、音乐剧的深度解析、评论、赏析类视频
2. 包含舞台艺术术语，如：舞台设计、布景、灯光设计、戏剧性张力、演员走位、舞台语言、音乐主题等
3. 贴合平台用户真实搜索习惯
4. 多样化，覆盖话剧、歌剧、音乐剧等不同演出形式

输出JSON:
{
    "keywords": ["关键词1", "关键词2", ...]
}""",
        "generate_keywords_user_prompt": """主题: {topic}

请生成{num_keywords}个最有效的搜索关键词，聚焦舞台表演艺术的 video essay / 赏析 / 评论类内容。
语言要求: {lang_rule}

必备关注点（可自由组合）：
- 舞台设计（Stage Design）：布景象征意义、空间调度（Blocking）、极简主义 vs 写实布景、道具符号学
- 舞台灯光（Theatrical Lighting）：色彩心理（暖光/冷光叙事）、追光（Follow Spot）、侧光与逆光的戏剧效果、灯光节奏与情绪
- 表演分析（Performance）：斯坦尼斯拉夫斯基体系、布莱希特间离效果（Verfremdungseffekt）、形体表演、歌剧唱腔与情感表达、音乐剧唱跳表演整合
- 导演与美学：导演概念（Director's Concept）、戏剧文本与舞台呈现的关系、经典复排分析
- 音乐分析：歌剧主导动机（Leitmotif）、音乐剧歌曲推动叙事的方式
- 可结合具体剧目案例（如：《等待戈多》的空舞台美学、《悲惨世界》的旋转舞台、瓦格纳乐剧）""",
        "judge_relevance_system_prompt": """你是一个舞台与现场表演艺术视频内容分析专家。你的任务是判断给定视频是否属于各类舞台/现场表演的分析、赏析、评论、解说或深度讨论。

涵盖范围（广义舞台表演）：
- 戏剧、歌剧、音乐剧（舞台设计、灯光、导演手法、剧目评论）
- 脱口秀/单口喜剧（stand-up comedy）：专场解说、喜剧技巧分析、comedian风格评论
- 小品/素描喜剧（sketch comedy）：SNL等节目分析、喜剧写作技巧
- 即兴喜剧（improv）：即兴表演技巧、剧团分析
- 芭蕾/现代舞/舞蹈剧场
- 歌舞表演秀（cabaret）、变装表演（drag show）
- 马戏/杂技（Cirque du Soleil等）、魔术表演
- 默剧、肢体喜剧、小丑表演
- 口语诗歌（spoken word / poetry slam）
- 脱口秀节目（talk show）的表演分析
- 各国传统戏曲（京剧、歌舞伎、能剧等）

判断标准：
- ✅ 符合：对上述任何舞台/现场表演形式进行解说、分析、评论、排名、深度讨论的视频
- ✅ 符合：喜剧专场的review/analysis、comedian技巧breakdown、音乐剧review
- ❌ 不符合：纯粹的演出录制（无解说分析）、纯音乐MV、电影分析（非舞台）、游戏/动漫内容、新闻/科普

输出JSON:
{
    "is_relevant": true/false,
    "reason": "判断理由",
    "confidence": 0.0-1.0
}""",
        "judge_relevance_user_hint": "请判断该视频是否聚焦舞台/现场表演艺术（含戏剧、音乐剧、脱口秀、喜剧、舞蹈、魔术等）的分析、解说或评论，并输出理由。",
        "variant_focus": "舞台表演、音乐剧、脱口秀喜剧、舞蹈和现场表演艺术分析",
        "fallback_map": {
            "en": [
                # --- Stand-up comedy & comedy specials ---
                "stand up comedy special analysis",
                "best stand up comedy specials ranked",
                "stand up comedy writing technique",
                "how stand up comedians craft jokes",
                "stand up comedy structure callback",
                "George Carlin comedy genius analysis",
                "Dave Chappelle stand up breakdown",
                "Bo Burnham Inside special analysis",
                "Hannah Gadsby Nanette comedy essay",
                "John Mulaney comedy style analysis",
                "Ali Wong stand up comedy review",
                "Hasan Minhaj storytelling comedy",
                "Trevor Noah comedy special review",
                "Kevin Hart comedy tour analysis",
                "Chris Rock comedy special breakdown",
                "Jerry Seinfeld observational comedy",
                "Eddie Murphy Raw Delirious analysis",
                "Richard Pryor stand up legacy",
                "Ricky Gervais comedy analysis",
                "Jimmy Carr comedy style dark humor",
                "Anthony Jeselnik comedy analysis",
                "Nate Bargatze comedy style review",
                "Taylor Tomlinson comedy special",
                "Sam Morril stand up review",
                "Mark Normand comedy analysis",
                "Shane Gillis comedy special review",
                "Matt Rife comedy special analysis",
                "Neal Brennan comedy blocks analysis",
                "Mike Birbiglia one man show",
                "Demetri Martin comedy style visual",
                "Mitch Hedberg one liner comedy genius",
                "stand up comedy Netflix special review",
                "best comedy specials 2024 ranked",
                "stand up comedy evolution history",
                "comedy club open mic culture",
                # --- Sketch comedy & improv ---
                "SNL sketch comedy analysis",
                "Saturday Night Live best sketches breakdown",
                "sketch comedy writing technique",
                "improv comedy rules explained",
                "Second City improv theater history",
                "UCB improv comedy training",
                "Whose Line Is It Anyway improv analysis",
                "Key and Peele sketch comedy genius",
                "Monty Python comedy analysis",
                "sketch comedy vs stand up comparison",
                "Kids in the Hall comedy analysis",
                "MadTV vs SNL sketch comparison",
                "Tim Robinson I Think You Should Leave",
                "A Show About Nothing Seinfeld analysis",
                "comedy duo chemistry analysis",
                "improv theater long form short form",
                # --- Musical theater ---
                "musical theater explained for beginners",
                "Broadway musical review analysis",
                "why musicals make you cry analysis",
                "best Broadway musicals ranked essay",
                "musical theater acting vs film acting",
                "Broadway show review 2024",
                "West End musical review analysis",
                "Sondheim musical genius explained",
                "Hamilton musical cultural impact",
                "Wicked musical deep dive analysis",
                "Les Miserables musical essay",
                "Phantom of the Opera legacy analysis",
                "Dear Evan Hansen musical review",
                "Hadestown musical mythology analysis",
                "Six musical history retelling",
                "Come From Away musical storytelling",
                "Beetlejuice musical review",
                "Moulin Rouge musical spectacle",
                "Chicago musical Fosse choreography",
                "Rent musical cultural impact essay",
                "Book of Mormon musical satire",
                "Sweeney Todd musical analysis",
                "Into the Woods fairy tale musical",
                "Company musical loneliness theme",
                "Cabaret musical political commentary",
                "Spring Awakening musical analysis",
                "Next to Normal musical mental health",
                "Waitress musical review analysis",
                "Mean Girls musical adaptation review",
                "Heathers musical dark comedy analysis",
                "Legally Blonde musical review",
                "Matilda musical Roald Dahl adaptation",
                "The Last Five Years musical structure",
                "Tick Tick Boom musical analysis",
                "Newsies musical choreography review",
                "Frozen musical Broadway adaptation",
                "Lion King musical staging puppetry",
                "Aladdin musical Broadway review",
                "Andrew Lloyd Webber musicals ranked",
                "Lin-Manuel Miranda musical genius",
                "musical theater choreography breakdown",
                "Tony Awards best musical analysis",
                "Broadway vs West End comparison",
                # --- Opera ---
                "opera explained for beginners",
                "opera staging modern interpretation",
                "Wagner Ring Cycle analysis essay",
                "Verdi opera dramatic analysis",
                "Puccini La Boheme analysis",
                "Mozart Magic Flute essay",
                "Carmen opera analysis review",
                "opera singer technique analysis",
                "opera house architecture design",
                "Metropolitan Opera behind the scenes",
                "opera vs musical theater comparison",
                # --- Drama & theater ---
                "theater directing techniques explained",
                "Shakespeare staging analysis essay",
                "Arthur Miller Death of Salesman analysis",
                "Tennessee Williams Streetcar analysis",
                "August Wilson plays analysis",
                "theater review critique essay",
                "immersive theater Sleep No More",
                "one person show monologue analysis",
                "devised theater collaborative creation",
                "verbatim theater documentary drama",
                "theater costume design symbolism",
                "stage design set design analysis",
                # --- Dance & ballet ---
                "ballet choreography analysis",
                "Swan Lake ballet interpretation",
                "contemporary dance theater analysis",
                "Pina Bausch dance theater essay",
                "hip hop dance battle analysis",
                "ballroom dance competition analysis",
                "Nutcracker ballet staging review",
                "dance performance storytelling",
                # --- Cabaret, drag, variety ---
                "cabaret performance art analysis",
                "drag show performance art essay",
                "RuPaul Drag Race performance analysis",
                "drag queen lip sync analysis",
                "burlesque show history analysis",
                "variety show vaudeville history",
                "talent show performance analysis",
                "America's Got Talent best acts analysis",
                "Britain's Got Talent performance review",
                # --- Circus & magic ---
                "Cirque du Soleil show analysis",
                "circus arts performance review",
                "magic show performance analysis",
                "Penn and Teller magic explained",
                "David Copperfield illusion analysis",
                "Derren Brown mentalism analysis",
                "street magic performance review",
                "acrobatics circus performance art",
                # --- Physical comedy & clown ---
                "physical comedy slapstick analysis",
                "clown performance art analysis",
                "mime performance Marcel Marceau",
                "Charlie Chaplin comedy genius analysis",
                "Buster Keaton physical comedy essay",
                "Mr Bean physical comedy analysis",
                "Jacques Tati comedy style essay",
                # --- Spoken word & poetry ---
                "spoken word poetry performance",
                "poetry slam competition analysis",
                "TED talk performance technique",
                "storytelling performance analysis",
                "oral storytelling tradition analysis",
                # --- Roast & panel comedy ---
                "comedy roast best moments analysis",
                "roast battle comedy analysis",
                "panel show comedy analysis",
                "late night talk show monologue analysis",
                "talk show host comedy style comparison",
                "Jimmy Fallon vs Jimmy Kimmel vs Colbert",
                "Conan O'Brien comedy style analysis",
                "Craig Ferguson late night genius",
                "Graham Norton Show best moments",
                # --- World theater traditions ---
                "kabuki theater visual analysis",
                "Beijing opera performance analysis",
                "Noh theater mask symbolism",
                "commedia dell arte performance",
                "Indian classical dance Bharatanatyam",
                "flamenco dance performance analysis",
                "African theater performance tradition",
                "puppetry theater art analysis",
                "shadow puppet theater analysis",
                "pantomime British theater tradition",
                # --- General performance analysis ---
                "best live performances ever analysis",
                "stage presence performance technique",
                "audience interaction performance",
                "live performance vs recorded comparison",
                "concert staging visual design",
                "award show performance analysis",
                "Super Bowl halftime show analysis",
                "Eurovision performance analysis",
                "festival performance stage design",
                "theater architecture design analysis",
            ],
            "zh-CN": [
                "舞台设计 布景象征 戏剧分析",
                "舞台灯光 色彩情绪 解析",
                "斯坦尼斯拉夫斯基 表演体系 讲解",
                "布莱希特 间离效果 戏剧论",
                "歌剧 主导动机 瓦格纳 分析",
                "音乐剧 歌曲叙事结构 视频论文",
                "话剧导演 舞台调度 美学",
                "歌剧咏叹调 情感表达 赏析",
                "极简舞台 空舞台 戏剧美学",
                "音乐剧 唱跳表演 深度解读",
                "汉密尔顿 音乐剧 深度解析",
                "悲惨世界 音乐剧 舞台分析",
                "歌剧魅影 舞台设计 赏析",
                "猫 音乐剧 经典分析",
                "芝加哥 音乐剧 舞蹈风格",
                "西区故事 音乐剧 编舞分析",
                "莎士比亚 戏剧 舞台呈现",
                "契诃夫 话剧 导演手法",
                "曹禺 雷雨 话剧分析",
                "赖声川 戏剧 创作分析",
                "孟京辉 先锋戏剧 解读",
                "开心麻花 话剧 喜剧分析",
                "京剧 表演艺术 赏析",
                "昆曲 舞台美学 解说",
                "越剧 表演 深度解读",
                "黄梅戏 艺术特色 分析",
                "话剧 表演技巧 深度讲解",
                "百老汇 音乐剧 发展史",
                "伦敦西区 音乐剧 对比分析",
                "歌剧 威尔第 普契尼 赏析",
                "芭蕾舞 天鹅湖 舞台赏析",
                "现代舞 舞蹈剧场 分析",
                "沉浸式戏剧 不眠之夜 解析",
                "戏剧节 乌镇 阿维尼翁",
                "音乐剧 声乐表演 分析",
                "舞台服装设计 角色塑造",
                "戏剧文学 剧本分析 解读",
                "独角戏 独白 表演分析",
                "肢体剧场 形体表演 赏析",
                "木偶戏 皮影戏 艺术分析",
            ],
            "zh-TW": [
                "舞台設計 佈景象徵 戲劇分析",
                "舞台燈光 色彩情緒 解析",
                "斯坦尼斯拉夫斯基 表演體系 講解",
                "布萊希特 間離效果 戲劇論",
                "歌劇 主導動機 瓦格納 分析",
                "音樂劇 歌曲敘事結構 視頻論文",
                "話劇導演 舞台調度 美學",
                "歌劇詠嘆調 情感表達 賞析",
                "極簡舞台 空舞台 戲劇美學",
                "音樂劇 唱跳表演 深度解讀",
                "漢密爾頓 音樂劇 深度解析",
                "悲慘世界 音樂劇 舞台分析",
                "歌劇魅影 舞台設計 賞析",
                "莎士比亞 戲劇 舞台呈現",
                "京劇 表演藝術 賞析",
                "歌仔戲 表演 深度解讀",
                "百老匯 音樂劇 發展史",
                "現代舞 舞蹈劇場 分析",
                "沉浸式戲劇 解析",
                "芭蕾舞 天鵝湖 舞台賞析",
            ],
        },
    },
    "game_arts": {
        "generate_keywords_system_prompt": """你是一个专注于游戏艺术领域的视频搜索专家。根据用户主题，生成高质量的搜索关键词。

关键词要求：
1. 聚焦于游戏视觉艺术、CG 动画、游戏美术设计的深度分析、评论、解说类视频
2. 包含游戏艺术术语，如：视觉叙事、关卡设计、美术风格、色彩指导、粒子特效、程序生成、CG 渲染、动作捕捉等
3. 贴合平台用户真实搜索习惯
4. 多样化，覆盖独立游戏、3A 大作、CG 过场动画等不同内容

输出JSON:
{
    "keywords": ["关键词1", "关键词2", ...]
}""",
        "generate_keywords_user_prompt": """主题: {topic}

请生成{num_keywords}个最有效的搜索关键词，聚焦游戏艺术的 video essay / 赏析 / 视觉分析类内容。
语言要求: {lang_rule}

必备关注点（可自由组合）：
- 互动视觉（Interactive Visuals）：视觉叙事（Visual Storytelling）与叙事环境设计（Environmental Storytelling）、关卡视觉引导（Visual Guidance）、UI/UX 美学、色彩指导（Color Direction）与情感
- CG 技术与美学：实时渲染（Real-time Rendering）vs 离线渲染、光线追踪（Ray Tracing）的视觉影响、PBR 材质美学、程序生成（Procedural Generation）的视觉逻辑
- 美术风格分析：像素艺术美学、手绘风格游戏（Cel Shading）、写实主义 3A 美术、印象派游戏画面分析
- 游戏 CG 动画：动作捕捉（Mocap）表演分析、过场动画镜头语言、CG 电影化叙事
- 具体案例：《风之旅人》的极简视觉叙事、《最后生还者》的环境叙事、《荒野大镖客2》的光影美学、《空洞骑士》的手绘美学""",
        "judge_relevance_system_prompt": """你是一个游戏艺术视频内容分析专家。你的任务是判断给定视频是否属于游戏视觉艺术（互动视觉设计、CG 动画）的深度分析、赏析、评论类 video essay。

判断标准：
- ✅ 符合：游戏美术风格分析、CG 动画技术与美学解析、游戏视觉叙事分析、关卡设计美学、游戏色彩指导评论、游戏 CG 电影化讨论
- ❌ 不符合：游戏攻略/实况（无艺术分析）、游戏性（Gameplay Mechanics）纯设计讨论（无视觉艺术角度）、游戏硬件评测、游戏配乐纯音乐分析（无视觉关联）

输出JSON:
{
    "is_relevant": true/false,
    "reason": "判断理由",
    "confidence": 0.0-1.0
}""",
        "judge_relevance_user_hint": "请判断该视频是否聚焦游戏视觉艺术（美术风格/CG动画/视觉叙事）的深度分析，并输出理由。",
        "variant_focus": "游戏视觉艺术、CG动画、美术风格和视觉叙事",
        "fallback_map": {
            "en": [
                "game visual storytelling environmental design essay",
                "video game color direction art analysis",
                "real-time rendering ray tracing visual aesthetic",
                "cel shading hand-drawn game art breakdown",
                "game CG cinematic animation analysis",
                "level design visual guidance art direction",
                "procedural generation visual logic game",
                "pixel art aesthetic indie game essay",
                "motion capture performance game cinematic",
                "AAA game realistic art style breakdown",
                "Elden Ring art direction dark fantasy analysis",
                "Hollow Knight hand drawn art style essay",
                "Journey game visual storytelling minimalism",
                "Last of Us environmental storytelling analysis",
                "Red Dead Redemption 2 lighting art essay",
                "Zelda Tears of the Kingdom art direction",
                "Breath of the Wild cel shading analysis",
                "God of War art direction Norse mythology",
                "Hades game art style supergiant analysis",
                "Cuphead animation art style 1930s cartoon",
                "Ori and the Blind Forest art analysis",
                "Gris game visual metaphor art essay",
                "Inside Limbo game art minimalism analysis",
                "Dark Souls visual design world building",
                "Bloodborne gothic art direction analysis",
                "Disco Elysium art style painting analysis",
                "Cyberpunk 2077 art direction neon aesthetic",
                "Final Fantasy CG cinematic evolution",
                "Metal Gear Solid cinematic storytelling",
                "Death Stranding visual design Kojima",
                "Shadow of the Colossus scale art design",
                "Ico minimalist game art analysis",
                "Okami Japanese art style game essay",
                "Persona 5 UI design art direction",
                "Stray game environment design analysis",
                "Returnal visual effects particle art",
                "Control brutalist architecture game design",
                "Bioshock art deco visual design essay",
                "Firewatch art style color palette analysis",
                "Monument Valley impossible geometry art",
                "game UI UX design aesthetic analysis",
                "GDC talk game art direction",
                "game concept art design process",
                "game environment art world building",
                "game character design visual analysis",
                "indie game art style unique aesthetic",
                "game lighting mood atmosphere design",
                "game animation principles breakdown",
                "game visual effects VFX breakdown",
                "game shader art stylized rendering",
                "Unreal Engine 5 Nanite visual showcase",
                "game art evolution PS1 to PS5",
                "retro game pixel art nostalgia essay",
                "game color theory mood storytelling",
                "game camera work cinematic framing",
                "Naughty Dog cinematic game design",
                "FromSoftware visual design philosophy",
                "Nintendo art direction design philosophy",
                "game world design open world aesthetic",
                "horror game visual design atmosphere",
                "Resident Evil art direction evolution",
                "Silent Hill visual symbolism analysis",
                "Alan Wake visual storytelling light dark",
                "It Takes Two visual design co-op game",
                "Sable game art style Moebius influence",
                "Tunic isometric art design analysis",
                "game typography font design analysis",
                "game poster key art visual analysis",
                "Valorant character design art breakdown",
                "League of Legends splash art analysis",
            ],
            "zh-CN": [
                "游戏视觉叙事 环境叙事 解析",
                "游戏美术 色彩指导 风格分析",
                "实时渲染 光线追踪 视觉美学",
                "卡通渲染 手绘风格 游戏美术赏析",
                "游戏 CG 过场动画 镜头语言",
                "关卡设计 视觉引导 美术方向",
                "程序生成 视觉逻辑 游戏美学",
                "像素艺术 独立游戏 美学分析",
                "动作捕捉 游戏电影化 表演解析",
                "3A游戏 写实美术 深度解说",
                "艾尔登法环 美术设计 暗黑奇幻",
                "空洞骑士 手绘美术 风格分析",
                "风之旅人 极简视觉 叙事设计",
                "最后生还者 环境叙事 美术解析",
                "荒野大镖客2 光影美学 分析",
                "塞尔达传说 美术风格 解读",
                "战神 美术设计 北欧神话",
                "哈迪斯 美术风格 超巨游戏",
                "茶杯头 动画美术 1930年代风格",
                "奥日与黑暗森林 美术赏析",
                "黑暗之魂 视觉设计 世界构建",
                "血源诅咒 哥特美术 风格分析",
                "极乐迪斯科 绘画风格 美术解析",
                "赛博朋克2077 美术方向 霓虹美学",
                "最终幻想 CG动画 进化史",
                "合金装备 电影化叙事 分析",
                "死亡搁浅 视觉设计 小岛秀夫",
                "旺达与巨像 尺度感 美术设计",
                "大神 日本画风 游戏美术",
                "女神异闻录5 UI设计 美术方向",
                "游戏UI设计 美学分析 赏析",
                "GDC 游戏美术 演讲 分享",
                "游戏概念设计 原画 创作过程",
                "游戏场景美术 世界构建 分析",
                "游戏角色设计 视觉分析 解说",
                "独立游戏 美术风格 独特美学",
                "游戏光影 氛围 设计分析",
                "虚幻引擎5 画面展示 技术美学",
                "游戏美术进化 PS1到PS5",
                "恐怖游戏 视觉设计 氛围营造",
                "生化危机 美术方向 进化史",
                "寂静岭 视觉象征 美术分析",
                "原神 美术设计 风格解析",
                "黑神话悟空 美术 视觉分析",
                "游戏色彩理论 情绪叙事",
                "游戏镜头语言 电影化构图",
                "任天堂 美术设计 设计哲学",
                "宫崎英高 视觉设计 哲学",
                "开放世界 美术设计 美学分析",
                "英雄联盟 原画 美术赏析",
            ],
            "zh-TW": [
                "遊戲視覺敘事 環境敘事 解析",
                "遊戲美術 色彩指導 風格分析",
                "即時渲染 光線追蹤 視覺美學",
                "卡通渲染 手繪風格 遊戲美術賞析",
                "遊戲 CG 過場動畫 鏡頭語言",
                "關卡設計 視覺引導 美術方向",
                "程序生成 視覺邏輯 遊戲美學",
                "像素藝術 獨立遊戲 美學分析",
                "動作捕捉 遊戲電影化 表演解析",
                "3A遊戲 寫實美術 深度解說",
                "艾爾登法環 美術設計 暗黑奇幻",
                "空洞騎士 手繪美術 風格分析",
                "薩爾達傳說 美術風格 解讀",
                "最後生還者 環境敘事 美術解析",
                "戰神 美術設計 北歐神話",
                "原神 美術設計 風格解析",
                "黑神話悟空 美術 視覺分析",
                "遊戲UI設計 美學分析 賞析",
                "恐怖遊戲 視覺設計 氛圍營造",
                "獨立遊戲 美術風格 獨特美學",
            ],
        },
    },
}

def normalize_search_keyword(keyword: str, max_length: int = 48) -> str:
    if not keyword:
        return ""

    normalized = str(keyword).strip()
    normalized = re.sub(r"（[^）]*）|\([^)]*\)|【[^】]*】|\[[^\]]*\]", "", normalized)

    for separator in ["：", ":", " - ", " — ", "｜", "|"]:
        if separator in normalized:
            head = normalized.split(separator, 1)[0].strip()
            if head:
                normalized = head
                break

    normalized = re.sub(r"[《》“”\"'`]", "", normalized)
    normalized = re.sub(r"[，,、/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if len(normalized) <= max_length:
        return normalized

    parts = [part for part in normalized.split(" ") if part]
    if len(parts) > 1:
        compact = []
        total = 0
        for part in parts:
            extra = len(part) if not compact else len(part) + 1
            if total + extra > max_length:
                break
            compact.append(part)
            total += extra
        if compact:
            return " ".join(compact)

    return normalized[:max_length].strip()


def normalize_search_keywords(keywords: List[str], num_keywords: int, max_length: int = 48) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        candidate = normalize_search_keyword(keyword, max_length=max_length)
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(candidate)
        if len(normalized) >= num_keywords:
            break
    return normalized


class LLMClient:
    """大模型API客户端，用于关键词生成和视频相关性判断"""

    def __init__(self, api_key: str, base_url: str = "https://jeniya.top"):
        self.api_key = api_key
        self.base_url = base_url

    def generate_search_keywords(self, topic: str, num_keywords: int = 5, language: str | None = None, category: str = "cinematic_arts") -> List[str]:
        """使用AI生成YouTube搜索关键词"""
        language_prompts = {
            "en": ("English", "请全部使用自然的英文关键词，避免任何中文字符。"),
            "zh-CN": ("简体中文", "请全部使用简体中文关键词，不要出现繁体或英文。"),
            "zh-TW": ("繁體中文", "請全部使用繁體中文關鍵詞，不要包含簡體或英文。"),
        }
        lang_label, lang_rule = language_prompts.get(language, ("多语言", "可灵活混合中英文关键词，根据语义最优选择。"))

        cfg = CATEGORY_CONFIGS[category]
        system_prompt = cfg["generate_keywords_system_prompt"]
        user_prompt = cfg["generate_keywords_user_prompt"].format(topic=topic, num_keywords=num_keywords, lang_rule=lang_rule)

        try:
            content = self._call_api(system_prompt, user_prompt)
            result = self._parse_response(content)
            keywords = result.get("keywords", [])
            if keywords:
                normalized_keywords = normalize_search_keywords(keywords, num_keywords)
                if normalized_keywords:
                    print(f"✅ AI生成{lang_label}关键词: {', '.join(normalized_keywords)}")
                    return normalized_keywords
        except Exception as exc:
            print(f"⚠️  AI生成关键词失败: {exc}")

        fallback_map = cfg["fallback_map"]
        fallback = fallback_map.get(language, [
            "cinematography shot size",
            "蒙太奇 剪辑 声音设计",
            "mise en scene blocking",
            "sound bridge diegetic",
        ])
        print(f"使用{lang_label}预设关键词: {', '.join(fallback[:num_keywords])}")
        return fallback[:num_keywords]

    def generate_keyword_variants(
        self,
        base_keywords: List[str],
        num_variants: int = 5,
        language: str | None = None,
        category: str = "cinematic_arts",
    ) -> List[str]:
        """根据已有关键词生成新的变体"""
        if not base_keywords:
            return []

        language_prompts = {
            "en": "请输出自然的英文关键词。",
            "zh-CN": "请输出简体中文关键词。",
            "zh-TW": "請輸出繁體中文關鍵詞。",
        }
        lang_rule = language_prompts.get(language, "语言可自适应，但需与主题一致。")

        system_prompt = """你是一个YouTube高级搜索专家。请根据用户提供的关键词，生成新的变体以覆盖更多相关视频。"""

        seed_list = "\n".join(f"- {kw}" for kw in base_keywords[-10:])
        variant_focus = CATEGORY_CONFIGS[category]["variant_focus"]
        user_prompt = f"""已有关键词：\n{seed_list}\n\n请基于这些词生成{num_variants}个全新且不重复的关键词，聚焦{variant_focus}。\n{lang_rule}\n"""

        try:
            content = self._call_api(system_prompt, user_prompt)
            result = self._parse_response(content)
            keywords = result.get("keywords") or result.get("variants") or result.get("new_keywords")
            if isinstance(keywords, list) and keywords:
                return keywords[:num_variants]
        except Exception as exc:
            print(f"⚠️  AI生成关键词变体失败: {exc}")

        # Fallback: simple suffix/prefix variations
        suffixes = [
            "analysis",
            "breakdown",
            "tutorial",
            "case study",
            "cinematography",
            "editing techniques",
            "mise en scene",
            "sound design",
        ]
        fallback = []
        fallback_lower = set()
        for kw in base_keywords:
            for suffix in suffixes:
                variant = f"{kw} {suffix}".strip()
                lower_variant = variant.lower()
                if lower_variant not in fallback_lower:
                    fallback.append(variant)
                    fallback_lower.add(lower_variant)
                if len(fallback) >= num_variants:
                    break
            if len(fallback) >= num_variants:
                break
        return fallback[:num_variants]

    def judge_video_relevance(self, video_info: dict, keywords: List[str], category: str = "cinematic_arts") -> tuple[bool, str, float]:
        """使用AI判断YouTube视频是否与主题相关"""
        cfg = CATEGORY_CONFIGS[category]
        keywords_str = ", ".join(keywords)
        system_prompt = cfg["judge_relevance_system_prompt"]

        user_prompt = f"""关键词: {keywords_str}

视频信息:
标题: {video_info.get('title', 'N/A')}
描述: {video_info.get('description', 'N/A')}
频道: {video_info.get('channel', 'N/A')}
播放量: {video_info.get('view_count', 'N/A')}

{cfg["judge_relevance_user_hint"]}"""

        try:
            content = self._call_api(system_prompt, user_prompt)
            result = self._parse_response(content)
            return (
                result.get("is_relevant", False),
                result.get("reason", ""),
                float(result.get("confidence", 0.0) or 0.0),
            )
        except Exception as exc:
            print(f"⚠️  AI判断失败: {exc}")
            safe_title = video_info.get("title") or ""
            safe_desc = video_info.get("description") or ""
            text = (safe_title + " " + safe_desc).lower()
            for kw in keywords:
                if kw.lower() in text:
                    return True, "关键词匹配（降级）", 0.5
            return False, "未匹配关键词（降级）", 0.3

    def _call_api(self, system_prompt: str, user_prompt: str, timeout: int | None = None, max_retries: int = 2) -> str:
        if timeout is None:
            timeout = int(os.environ.get("SCRAPING_LLM_TIMEOUT", "60"))
        parsed = urlparse(self.base_url)
        host = parsed.netloc
        scheme = parsed.scheme
        path_prefix = parsed.path.rstrip("/")
        endpoint_path = f"{path_prefix}/v1/chat/completions"

        payload = json.dumps(
            {
                "model": "gpt-5-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
        )

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(max_retries):
            conn = None
            try:
                if attempt > 0:
                    print(f"重试 {attempt}/{max_retries - 1}...")
                    time.sleep(2)

                conn = (
                    http.client.HTTPSConnection(host, timeout=timeout)
                    if scheme == "https"
                    else http.client.HTTPConnection(host, timeout=timeout)
                )
                conn.request("POST", endpoint_path, payload, headers)
                res = conn.getresponse()
                data = res.read()
                response_text = data.decode("utf-8")

                if res.status != 200:
                    raise RuntimeError(f"API状态码 {res.status}: {response_text[:200]}")

                response_json = json.loads(response_text)
                choices = response_json.get("choices") or []
                if choices:
                    return choices[0]["message"]["content"]
                raise RuntimeError(f"API返回格式异常: {response_text[:200]}")
            except Exception as exc:
                last_error = exc
                print(f"❌ API调用失败 (尝试 {attempt + 1}/{max_retries}): {exc}")
            finally:
                if conn:
                    conn.close()

        raise RuntimeError(f"API调用失败: {last_error}")

    @staticmethod
    def _parse_response(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"```\s*$", "", content)

        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}


class SeleniumYouTubeCrawler:
    """使用Selenium模拟浏览器获取YouTube搜索结果"""

    LANGUAGE_PARAMS = {
        "en": {"hl": "en", "gl": "US"},
        "zh-CN": {"hl": "zh-CN", "gl": "CN"},
        "zh-TW": {"hl": "zh-TW", "gl": "TW"},
    }

    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    ]

    def __init__(self, driver_path: str | None = None, headless: bool = True, wait_seconds: float = 2.0):
        if webdriver is None or Options is None or Service is None:
            raise ImportError("selenium is not installed; rerun without --use-selenium or install selenium")
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--lang=en-US")
        chrome_options.add_argument(f"--user-agent={random.choice(self._USER_AGENTS)}")
        if headless:
            chrome_options.add_argument("--headless=new")

        service = Service(driver_path) if driver_path else Service()
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except WebDriverException as exc:
            raise RuntimeError("无法启动Chrome WebDriver，请检查chromedriver路径或安装情况") from exc

        self.wait_seconds = wait_seconds

    def close(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _build_search_url(self, keyword: str, language_code: str | None = None) -> str:
        params = self.LANGUAGE_PARAMS.get(language_code, {"hl": "en", "gl": "US"})
        hl = params.get("hl", "en")
        gl = params.get("gl", "US")
        return f"https://www.youtube.com/results?search_query={quote_plus(keyword)}&hl={hl}&gl={gl}"

    def search(self, keyword: str, language_code: str | None = None, scroll_times: int = 3) -> list[dict]:
        url = self._build_search_url(keyword, language_code)
        print(f"Selenium抓取: {url}")
        self.driver.get(url)
        time.sleep(self.wait_seconds)
        time.sleep(random.uniform(0.5, 2.0))

        collected: dict[str, dict] = {}
        for _ in range(max(scroll_times, 1)):
            html = self.driver.page_source
            soup = BeautifulSoup(html, "lxml")
            anchors = soup.select("a#video-title, a#video-title-link, a.yt-simple-endpoint.style-scope.ytd-video-renderer")

            for anchor in anchors:
                href = anchor.get("href")
                if not href or "/watch" not in href:
                    continue
                video_id = None
                if "v=" in href:
                    video_id = href.split("v=")[-1].split("&")[0]
                elif href.startswith("/watch/"):
                    video_id = href.split("/watch/")[-1]
                if not video_id:
                    continue
                if video_id in collected:
                    continue
                title = anchor.get("title") or anchor.get_text(strip=True)
                collected[video_id] = {
                    "id": video_id,
                    "title": title,
                    "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel": None,
                }

            self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(self.wait_seconds)

        print(f"Selenium抓取关键词 '{keyword}' 获得 {len(collected)} 条结果")
        return list(collected.values())

    def collect_for_keywords(
        self,
        keywords: Iterable[str],
        language_code: str | None = None,
        scroll_times: int = 3,
    ) -> list[dict]:
        aggregated: list[dict] = []
        seen_ids: set[str] = set()
        for keyword in keywords:
            results = self.search(keyword, language_code=language_code, scroll_times=scroll_times)
            for video in results:
                vid = video.get("id")
                if not vid or vid in seen_ids:
                    continue
                seen_ids.add(vid)
                aggregated.append(video)
        return aggregated


class YouTubeDownloader:
    """YouTube视频搜索与下载器"""

    def __init__(
        self,
        output_dir: str = str(VIDEO_YOUTUBE_ROOT),
        format_selection: str = "bestvideo+bestaudio/best",
        min_confidence: float = 0.55,
        cookies_file: str | None = None,
        cookies_browser: str | None = None,
        cookies_browser_profile: str | None = None,
        cookies_browser_keyring: str | None = None,
        cookies_browser_container: str | None = None,
        category: str = "cinematic_arts",
    ):
        self.category = category
        self.output_dir = Path(output_dir) / category
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format_selection = format_selection

        # 记录文件默认在 project_paths.TRACKING_ROOT 下，也可用环境变量覆写到新结构
        tracking_root = Path(os.environ.get("AUDIOVISUAL_BENCH_TRACKING_ROOT", str(TRACKING_ROOT)))
        record_dir = tracking_root / "youtube" / category
        record_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_record_file = record_dir / "downloaded_video_ids.txt"

        # Migration: copy old tracking files to new location
        if not self.downloaded_record_file.exists():
            import shutil
            for old in [
                record_dir / ".downloaded_video_ids.txt",
                record_dir / "downloaded_video_ids.txt",
                Path(output_dir) / ".downloaded_video_ids.txt",
                Path(output_dir) / "downloaded_video_ids.txt",
                Path(output_dir) / category / ".downloaded_video_ids.txt",
                Path(output_dir) / category / "downloaded_video_ids.txt",
            ]:
                if old.exists():
                    shutil.copy2(old, self.downloaded_record_file)
                    break
        self.tracked_ids = self._load_downloaded_records()
        self.downloaded_ids = self._load_existing_downloaded_ids(Path(output_dir))
        self.download_lock = threading.Lock()
        self.min_confidence = min_confidence
        self.cookies_file = Path(cookies_file).expanduser() if cookies_file else None
        if self.cookies_file and not self.cookies_file.exists():
            raise FileNotFoundError(f"Cookies文件不存在: {self.cookies_file}")
        self.cookies_browser = cookies_browser
        self.cookies_browser_profile = cookies_browser_profile
        self.cookies_browser_keyring = cookies_browser_keyring
        self.cookies_browser_container = cookies_browser_container

    def _load_downloaded_records(self) -> set[str]:
        if self.downloaded_record_file.exists():
            try:
                with open(self.downloaded_record_file, "r", encoding="utf-8") as file:
                    return {line.strip() for line in file if line.strip()}
            except OSError as exc:
                print(f"⚠️  加载下载记录失败: {exc}")
        return set()

    def _candidate_scan_roots(self, output_dir_root: Path) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()

        raw_roots = [str(output_dir_root)]
        env_roots = os.environ.get("AUDIOVISUAL_BENCH_EXISTING_VIDEO_ROOTS", "")
        if env_roots:
            raw_roots.extend(part for part in env_roots.split(os.pathsep) if part.strip())

        for raw_root in raw_roots:
            root = Path(raw_root).expanduser()
            try:
                resolved = root.resolve()
            except OSError:
                resolved = root
            if resolved in seen or not root.exists():
                continue
            seen.add(resolved)
            roots.append(root)
        return roots

    def _iter_existing_category_dirs(self, root: Path) -> list[Path]:
        candidates = [
            root / self.category,
            root / "youtube" / self.category,
            root / "youtube_videos" / self.category,
        ]
        platform_parents: list[Path] = []
        if root.name in {"youtube", "youtube_videos"}:
            platform_parents.append(root)
        for platform_dir_name in ["youtube", "youtube_videos"]:
            platform_dir = root / platform_dir_name
            if platform_dir.is_dir():
                platform_parents.append(platform_dir)

        for platform_dir in platform_parents:
            for child in platform_dir.iterdir():
                if child.is_dir() and child.name != "metadata":
                    candidates.append(child)

        if root.is_dir() and any(
            child.is_file() and child.suffix.lower() in VIDEO_FILE_EXTENSIONS
            for child in root.iterdir()
        ):
            candidates.append(root)

        resolved_dirs: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            resolved_dirs.append(candidate)
        return resolved_dirs

    def _load_existing_downloaded_ids(self, output_dir_root: Path) -> set[str]:
        existing_ids: set[str] = set()

        for root in self._candidate_scan_roots(output_dir_root):
            for category_dir in self._iter_existing_category_dirs(root):
                metadata_dir = category_dir / "metadata"
                if metadata_dir.is_dir():
                    for meta_path in metadata_dir.glob("*.json"):
                        try:
                            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            continue
                        source_id = metadata.get("source_id")
                        if source_id:
                            existing_ids.add(source_id)

                for video_path in category_dir.iterdir():
                    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_FILE_EXTENSIONS:
                        continue
                    stem = video_path.stem
                    if "__" not in stem:
                        continue
                    source_id, _ = stem.split("__", 1)
                    source_id = source_id.strip()
                    if source_id:
                        existing_ids.add(source_id)

        return existing_ids

    def _save_downloaded_record(self, video_id: str) -> None:
        with self.download_lock:
            if video_id in self.downloaded_ids:
                return
            self.downloaded_ids.add(video_id)
            try:
                with open(self.downloaded_record_file, "a", encoding="utf-8") as file:
                    file.write(f"{video_id}\n")
            except OSError as exc:
                print(f"⚠️  写入下载记录失败: {exc}")

    def is_downloaded(self, video_id: str) -> bool:
        with self.download_lock:
            return video_id in self.downloaded_ids

    def search_videos(self, keyword: str, page: int = 1, page_size: int = 20) -> list[dict]:
        query_size = page * page_size
        search_spec = f"ytsearch{query_size}:{keyword}"
        opts = self._build_ydl_options(
            {
                "quiet": True,
                "extract_flat": True,
                "skip_download": True,
            }
        )
        ssl_retries = 3
        for attempt in range(ssl_retries + 1):
            try:
                with YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(search_spec, download=False)
                break
            except DownloadError as exc:
                if "SSL" in str(exc) and attempt < ssl_retries:
                    wait = 60 * (attempt + 1)
                    print(f"⚠️ SSL错误，等待{wait}秒后重试 ({attempt+1}/{ssl_retries})...")
                    time.sleep(wait)
                    continue
                print(f"❌ 搜索失败: {exc}")
                return []

        entries = info.get("entries") or []
        start = (page - 1) * page_size
        return entries[start : start + page_size]

    def download_video(self, video: dict, quality: str = "bestvideo+bestaudio/best") -> Path:
        video_url = (
            video.get("webpage_url")
            or video.get("url")
            or (f"https://www.youtube.com/watch?v={video.get('id')}" if video.get("id") else None)
        )
        if not video_url:
            raise ValueError("无法解析视频链接")

        # Cap at 720p and 500MB to avoid multi-GB downloads blocking the queue
        size_capped_format = (
            "bestvideo[height<=720][filesize<500M]+bestaudio[filesize<100M]"
            "/bestvideo[height<=720]+bestaudio"
            "/best[height<=720]"
            "/best"
        )
        ydl_opts = self._build_ydl_options(
            {
                "outtmpl": str(self.output_dir / "%(id)s__%(title)s.%(ext)s"),
                "format": size_capped_format,
                "merge_output_format": "mp4",
                "writesubtitles": False,
                "quiet": False,
                "noprogress": False,
                "restrictfilenames": False,
                "ignoreerrors": False,
            }
        )

        print(f"开始下载: {video.get('title', video.get('id'))}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        output_path = self._find_downloaded_file(video.get("id"), video.get("title"))
        if not output_path:
            raise FileNotFoundError("未找到下载后的MP4文件")
        return output_path

    def _find_downloaded_file(self, video_id: str | None, title: str | None) -> Path | None:
        if video_id:
            matches = sorted(self.output_dir.glob(f"{video_id}__*.mp4"))
            if matches:
                return matches[-1]
        if title:
            pattern = re.sub(r"[\\/:*?\"<>|]+", "*", title)
            pattern = re.sub(r"\*+", "*", f"*__{pattern}*.mp4")
            matches = sorted(self.output_dir.glob(pattern))
            if matches:
                return matches[-1]
        return None

    def _build_ydl_options(self, extra_opts: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compose shared yt-dlp options so cookies apply everywhere."""
        opts: dict[str, Any] = {"remote_components": ["ejs:github"]}
        if self.cookies_file:
            opts["cookiefile"] = str(self.cookies_file)
        elif self.cookies_browser:
            opts["cookiesfrombrowser"] = (
                self.cookies_browser,
                self.cookies_browser_profile,
                self.cookies_browser_keyring,
                self.cookies_browser_container,
            )
        if extra_opts:
            opts.update(extra_opts)
        return opts

    def _download_video_task(
        self,
        video: dict,
        keywords: list[str],
        llm_client: LLMClient | None,
        quality: str,
        task_id: int,
        total_tasks: int,
    ) -> tuple[bool, str, str]:
        video_id = video.get("id") or video.get("url") or ""
        title = video.get("title", "未命名视频")
        channel = video.get("uploader", video.get("channel"))

        print(f"\n{'='*60}")
        print(f"[任务 {task_id}/{total_tasks}] {title}")
        print(f"频道: {channel}")
        print(f"ID: {video_id}")

        if not video_id:
            print("❌ 缺少视频ID，跳过")
            return False, "", title

        if self.is_downloaded(video_id):
            print("⏭️  已下载，跳过")
            return False, video_id, title

        if llm_client:
            print("使用AI判断相关性...")
            is_relevant, reason, confidence = llm_client.judge_video_relevance(
                {
                    "title": title,
                    "description": video.get("description", ""),
                    "channel": channel,
                    "view_count": video.get("view_count"),
                },
                keywords,
                category=self.category,
            )
            print(f"判断: {'✅ 相关' if is_relevant else '❌ 不相关'} | 置信度: {confidence:.2f}")
            print(f"理由: {reason}")
            if not is_relevant or confidence < self.min_confidence:
                return False, video_id, title

        try:
            self.download_video(video, quality=quality)
            self._save_downloaded_record(video_id)
            print(f"✅ 下载完成: {title}")
            return True, video_id, title
        except Exception as exc:
            print(f"❌ 下载失败[{title}]: {exc}")
            return False, video_id, title

    def search_and_download(
        self,
        keywords: Iterable[str],
        max_videos: int = 50,
        llm_client: LLMClient | None = None,
        quality: str = "bestvideo+bestaudio/best",
        max_workers: int = 2,
        max_pages_per_keyword: int = 5,
        language_code: str | None = None,
        variant_batch_size: int = 5,
    ) -> None:
        keywords_list = [kw for kw in keywords if kw]
        if not keywords_list:
            print("❌ 没有有效关键词")
            return

        keyword_pages = {kw: 1 for kw in keywords_list}
        exhausted_keywords: set[str] = set()
        seen_ids: set[str] = set()
        downloaded_count = 0
        round_idx = 1

        while downloaded_count < max_videos:
            active_keywords = [kw for kw in keywords_list if kw not in exhausted_keywords]
            if not active_keywords:
                if downloaded_count >= max_videos:
                    break
                if llm_client is None:
                    print("关键词全部耗尽，且无法生成新的变体。")
                    break

                print("\n🔄 关键词已耗尽，向AI请求新的变体...")
                variant_candidates = llm_client.generate_keyword_variants(
                    keywords_list,
                    num_variants=max(variant_batch_size, 1),
                    language=language_code,
                    category=self.category,
                )
                variant_candidates = [kw.strip() for kw in variant_candidates if kw and kw.strip()]
                new_keywords = [kw for kw in variant_candidates if kw not in keyword_pages]

                if not new_keywords:
                    print("⚠️ AI未能提供新的关键词，结束搜索。")
                    break

                print(f"✅ 新增关键词: {', '.join(new_keywords)}")
                for kw in new_keywords:
                    keywords_list.append(kw)
                    keyword_pages[kw] = 1
                continue

            print(f"\n{'='*60}")
            print(f"第 {round_idx} 轮搜索 | 已下载 {downloaded_count}/{max_videos}")
            print(f"{'='*60}")
            round_idx += 1

            for keyword in active_keywords:
                if downloaded_count >= max_videos:
                    break

                time.sleep(random.uniform(1.0, 3.0))
                page = keyword_pages[keyword]
                if page > max_pages_per_keyword:
                    exhausted_keywords.add(keyword)
                    continue

                print(f"\n--- 关键词: {keyword} | 第 {page} 页 ---")
                videos = self.search_videos(keyword, page=page, page_size=25)

                if not videos:
                    exhausted_keywords.add(keyword)
                    print("该关键词无更多结果，标记为耗尽")
                    continue

                keyword_pages[keyword] += 1
                new_videos = []
                for video in videos:
                    video_id = video.get("id") or video.get("url")
                    if not video_id or video_id in seen_ids:
                        continue
                    seen_ids.add(video_id)
                    new_videos.append(video)

                if not new_videos:
                    print("本页均已处理，继续下一页")
                    continue

                print(f"发现 {len(new_videos)} 个新视频，启动下载...")
                target_met = False
                for batch_start in range(0, len(new_videos), max_workers):
                    if downloaded_count >= max_videos:
                        target_met = True
                        break

                    batch = new_videos[batch_start:batch_start + max_workers]
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_video = {}
                        for offset, video in enumerate(batch, 1):
                            task_idx = batch_start + offset
                            future = executor.submit(
                                self._download_video_task,
                                video,
                                keywords_list,
                                llm_client,
                                quality,
                                task_idx,
                                len(new_videos),
                            )
                            future_to_video[future] = video

                        for future in as_completed(future_to_video):
                            success, video_id, title = future.result()
                            if success:
                                downloaded_count += 1
                                print(f"📊 进度: {downloaded_count}/{max_videos}")
                                if downloaded_count >= max_videos:
                                    target_met = True

                    if target_met:
                        break

        print(f"\n{'='*60}")
        if downloaded_count >= max_videos:
            print(f"🎉 完成任务，共下载 {downloaded_count} 个视频")
        else:
            print(f"⚠️ 仅下载 {downloaded_count}/{max_videos} 个视频，关键词已耗尽")
        print(f"{'='*60}")

    def download_from_video_list(
        self,
        videos: list[dict],
        keywords: Iterable[str],
        llm_client: LLMClient | None = None,
        quality: str = "bestvideo+bestaudio/best",
        max_videos: int = 10,
    ) -> None:
        keywords_list = list(keywords)
        if not videos:
            print("❌ Selenium未返回可下载的视频")
            return

        print(f"\n{'='*60}")
        print(f"Selenium提供 {len(videos)} 个候选视频，目标下载 {max_videos}")
        print(f"{'='*60}")

        downloaded_count = 0
        for idx, video in enumerate(videos, 1):
            if downloaded_count >= max_videos:
                break
            success, video_id, title = self._download_video_task(
                video,
                keywords_list,
                llm_client,
                quality,
                idx,
                len(videos),
            )
            if success:
                downloaded_count += 1
                print(f"📊 进度: {downloaded_count}/{max_videos}")

        print(f"\n{'='*60}")
        print(f"Selenium流程完成，共下载 {downloaded_count}/{max_videos} 个视频")
        print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube智能拉片视频下载器")
    parser.add_argument("--max", type=int, default=20, help="每种语言最多下载视频数量")
    parser.add_argument("--quality", type=str, default="bestvideo+bestaudio/best", help="yt-dlp格式表达式")
    parser.add_argument("--workers", type=int, default=3, help="线程并发数量")
    parser.add_argument("--topic", type=str, default="电影拉片与视听语言解析", help="关键词主题")
    parser.add_argument("--output", type=str, default=str(VIDEO_YOUTUBE_ROOT), help="输出目录")
    parser.add_argument("--api-key", type=str, default=os.environ.get("YOUTUBE_API_KEY") or os.environ.get("BILIBILI_API_KEY") or os.environ.get("API_KEY"), help="LLM API Key")
    parser.add_argument("--base-url", type=str, default=os.environ.get("YOUTUBE_BASE_URL") or os.environ.get("BILIBILI_BASE_URL") or (f"https://{os.environ['API_HOST']}" if os.environ.get("API_HOST") else "https://jeniya.top"), help="LLM Base URL")
    parser.add_argument("--cookies-file", type=str, default=None, help="yt-dlp cookies.txt路径 (Netscape格式)")
    parser.add_argument(
        "--cookies-browser",
        type=str,
        default=None,
        help="直接从浏览器读取cookies (chrome/brave/edge/firefox/safari等)",
    )
    parser.add_argument("--cookies-browser-profile", type=str, default=None, help="浏览器Profile名称，如 Default")
    parser.add_argument("--cookies-browser-keyring", type=str, default=None, help="浏览器cookies密钥ring，可选")
    parser.add_argument("--cookies-browser-container", type=str, default=None, help="Firefox容器ID，可选")
    parser.add_argument("--languages", type=str, default="en,zh-CN,zh-TW", help="需要运行的语言列表，逗号分隔，如 en,zh-CN,zh-TW")
    parser.add_argument("--num-keywords", type=int, default=6, help="每种语言生成的关键词数量")
    parser.add_argument("--use-selenium", action="store_true", help="使用Selenium进行动态爬虫")
    parser.add_argument("--chrome-driver", type=str, default=None, help="chromedriver路径，不传则使用系统默认")
    parser.add_argument("--selenium-scrolls", type=int, default=3, help="Selenium每个关键词下滑次数")
    parser.add_argument("--selenium-wait", type=float, default=2.0, help="Selenium每次滚动等待秒数")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="AI判定相关性的最低置信度")
    parser.add_argument("--max-pages", type=int, default=5, help="每个关键词最多翻页次数（非Selenium模式）")
    parser.add_argument("--show-browser", action="store_true", help="Selenium调试模式，显示真实浏览器窗口")
    parser.add_argument("--category", type=str, choices=list(CATEGORY_CONFIGS.keys()), default="cinematic_arts", help="视频类别")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    language_labels = {
        "en": "English",
        "zh-CN": "简体中文",
        "zh-TW": "繁體中文",
    }

    print("=" * 60)
    print("YouTube 智能视频下载器")
    print("=" * 60)
    print("🎥 当前类别: " + args.category)
    print(f"🌐 多语言流程: {', '.join(languages)}")
    print(f"📹 每种语言目标: {args.max} 个视频")
    print(f"⚙️  并发线程: {args.workers}")
    if args.use_selenium:
        print("🕹️  模式: Selenium动态爬虫")
    else:
        print("🕹️  模式: yt-dlp关键词搜索")

    print("\n初始化AI客户端...")
    llm_client = LLMClient(api_key=args.api_key, base_url=args.base_url)

    downloader = YouTubeDownloader(
        output_dir=args.output,
        min_confidence=args.min_confidence,
        cookies_file=args.cookies_file,
        cookies_browser=args.cookies_browser,
        cookies_browser_profile=args.cookies_browser_profile,
        cookies_browser_keyring=args.cookies_browser_keyring,
        cookies_browser_container=args.cookies_browser_container,
        category=args.category,
    )

    selenium_crawler = None
    if args.use_selenium:
        print("\n启动Selenium浏览器...")
        selenium_crawler = SeleniumYouTubeCrawler(
            driver_path=args.chrome_driver,
            headless=not args.show_browser,
            wait_seconds=args.selenium_wait,
        )

    try:
        for lang in languages:
            lang_label = language_labels.get(lang, lang)
            print(f"\n{'='*60}")
            print(f"▶️  当前语言: {lang_label} ({lang})")
            print(f"主题: {args.topic}")

            keywords = llm_client.generate_search_keywords(
                args.topic,
                num_keywords=args.num_keywords,
                language=lang,
                category=args.category,
            )

            # Put curated fallback keywords FIRST (shorter, more targeted),
            # then append AI-generated keywords after
            fallback_map = CATEGORY_CONFIGS[args.category].get("fallback_map", {})
            fallback_kws = fallback_map.get(lang, [])
            seen_lower = {kw.lower() for kw in fallback_kws}
            merged = list(fallback_kws)
            for kw in keywords:
                if kw.lower() not in seen_lower:
                    merged.append(kw)
                    seen_lower.add(kw.lower())
            keywords = merged
            print(f"合并后关键词总数: {len(keywords)} (预设 {len(fallback_kws)} + AI {args.num_keywords})")

            if args.use_selenium and selenium_crawler:
                selenium_results = selenium_crawler.collect_for_keywords(
                    keywords,
                    language_code=lang,
                    scroll_times=args.selenium_scrolls,
                )
                downloader.download_from_video_list(
                    selenium_results,
                    keywords=keywords,
                    llm_client=llm_client,
                    quality=args.quality,
                    max_videos=args.max,
                )
            else:
                downloader.search_and_download(
                    keywords=keywords,
                    max_videos=args.max,
                    llm_client=llm_client,
                    quality=args.quality,
                    max_workers=args.workers,
                    max_pages_per_keyword=args.max_pages,
                    language_code=lang,
                )
    finally:
        if selenium_crawler:
            selenium_crawler.close()


if __name__ == "__main__":
    main()
