#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilibili视频爬取工具
支持通过BV号或完整URL下载B站视频
支持关键词搜索和AI判断视频相关性
"""

import requests
import re
import json
import os
import http.client
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import argparse

REPO_ROOT = Path(__file__).resolve().parent
TRACKING_ROOT = Path(os.environ.get("TRACKING_ROOT", REPO_ROOT / "tracking"))
VIDEO_BILIBILI_ROOT = Path(os.environ.get("VIDEO_BILIBILI_ROOT", REPO_ROOT / "bilibili_videos"))


VIDEO_FILE_EXTENSIONS = {".mp4", ".mkv", ".webm", ".flv", ".mov", ".m4v"}

CATEGORY_CONFIGS = {
    "cinematic_arts": {
        "generate_keywords_system_prompt": """你是一个B站电影内容搜索专家。根据用户主题，生成高质量的搜索关键词。

关键词要求：
1. 聚焦于电影拉片、电影解说、视听语言、导演镜头技法
2. 包含电影技法术语，如：蒙太奇、镜头语言、场面调度、长镜头、希区柯克、色彩构图等
3. 贴合B站用户真实搜索习惯，兼顾中英组合（如："film breakdown 拉片"）
4. 多样化，覆盖不同电影流派与导演风格

输出JSON:
{
    "keywords": ["关键词1", "关键词2", ...]
}""",
        "generate_keywords_user_prompt": """主题: {topic}

请生成{num_keywords}个最有效的B站搜索关键词。尽量覆盖拉片、解说和电影技法。
语言要求: {lang_rule}

必备关注点（可自由组合）：
- 摄影（Cinematography）：镜头景别（Long Shot/Extreme Close-up）、机位角度（Dutch Angle/Bird's Eye）、运动方式（Dolly Zoom/Tracking）、光影色彩（High Key/Low Key，色彩心理）、构图法则（Rule of Thirds/Leading Lines）。
- 剪辑（Editing）：蒙太奇、长镜头、跳切、平行剪辑、闪回、转场（Match Cut/Smash Cut）、库里肖夫效应。
- 电影场面调度（Mise-en-scène）：场景与道具符号、人物走位与调度、服化道呈现人物关系。
- 声音（Sound Design）：声画关系（同步/反差）、叙事来源（Diegetic vs Non-diegetic）、声音桥、环境声、利用静默。
- 可结合导演或大师案例解释这些技法。""",
        "judge_relevance_system_prompt": """你是一个视频内容分析专家。你的任务是判断给定的B站视频是否与用户提供的关键词相关。

请仔细分析视频的标题、描述等信息，判断该视频是否真正属于用户关心的类别。

输出格式为JSON:
{
    "is_relevant": true/false,
    "reason": "判断理由",
    "confidence": 0.0-1.0
}""",
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
        "judge_relevance_system_prompt": """你是一个舞台表演艺术视频内容分析专家。你的任务是判断给定视频是否属于戏剧、歌剧、音乐剧的深度分析、赏析、评论或批评类 video essay。

判断标准：
- ✅ 符合：对舞台设计、灯光设计、表演艺术、戏剧文本、歌剧咏叹调、音乐剧选段进行深度解析；戏剧导演手法分析；剧目评论
- ❌ 不符合：演出宣传片/预告片、纯粹的表演记录（无解说分析）、声乐技巧练习教程、歌剧混剪娱乐向视频

输出JSON:
{
    "is_relevant": true/false,
    "reason": "判断理由",
    "confidence": 0.0-1.0
}""",
        "fallback_map": {
            "en": [
                "stage design symbolism theater essay",
                "theatrical lighting color mood analysis",
                "stanislavski method acting breakdown",
                "brechtian theater alienation effect explained",
                "opera leitmotif wagner music analysis",
                "musical theater song narrative structure",
                "blocking stage space director concept",
                "opera aria emotional expression analysis",
                "theater set design minimalism realism",
                "musical theater dance integration analysis",
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
            ],
        },
    },
}


class LLMClient:
    """大模型API客户端，用于判断视频相关性"""
    
    def __init__(self, api_key, base_url="https://jeniya.top"):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        
    def generate_search_keywords(self, topic, num_keywords=5, language=None, category="cinematic_arts"):
        """
        使用AI生成搜索关键词

        Args:
            topic: 主题描述
            num_keywords: 生成关键词数量
            language: 关键词语言（如 en/zh-CN/zh-TW）
            category: 内容类别

        Returns:
            关键词列表
        """
        language_prompts = {
            "en": ("English", "请全部使用自然的英文关键词，避免任何中文字符。"),
            "zh-CN": ("简体中文", "请全部使用简体中文关键词，不要出现繁体或英文。"),
            "zh-TW": ("繁體中文", "請全部使用繁體中文關鍵詞，不要包含簡體或英文。"),
        }
        lang_label, lang_rule = language_prompts.get(language, ("多语言", "可灵活混合中英文关键词，根据语义最优选择。"))

        config = CATEGORY_CONFIGS[category]
        system_prompt = config["generate_keywords_system_prompt"]
        user_prompt = config["generate_keywords_user_prompt"].format(topic=topic, num_keywords=num_keywords, lang_rule=lang_rule)
        
        try:
            content = self._call_api(system_prompt, user_prompt)
            result = self._parse_response(content)
            keywords = result.get('keywords', [])
            if keywords:
                normalized_keywords = normalize_search_keywords(keywords, num_keywords, category=category)
                if normalized_keywords:
                    print(f"✅ AI生成{lang_label}关键词: {', '.join(normalized_keywords)}")
                    return normalized_keywords
        except Exception as e:
            print(f"\n⚠️  AI生成关键词失败: {e}")
            print("使用预设关键词继续...")
        
        fallback_map = config["fallback_map"]
        fallback = fallback_map.get(language, [
            "cinematography shot size",
            "蒙太奇 剪辑 声音设计",
            "mise en scene blocking",
            "sound bridge diegetic",
        ])
        print(f"使用{lang_label}预设关键词: {', '.join(fallback[:num_keywords])}")
        return fallback[:num_keywords]
    
    def judge_video_relevance(self, video_info, keywords, category="cinematic_arts"):
        """
        使用大模型判断视频是否与关键词相关

        Args:
            video_info: 视频信息字典（包含title, desc等）
            keywords: 搜索关键词列表或字符串
            category: 内容类别

        Returns:
            (is_relevant: bool, reason: str, confidence: float)
        """
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
        else:
            keywords_str = str(keywords)

        system_prompt = CATEGORY_CONFIGS[category]["judge_relevance_system_prompt"]
        
        user_prompt = f"""关键词: {keywords_str}

视频信息:
标题: {video_info.get('title', 'N/A')}
描述: {video_info.get('desc', 'N/A')}
UP主: {video_info.get('owner', 'N/A')}
播放量: {video_info.get('play', 'N/A')}

请判断这个视频是否与关键词相关，并给出理由。"""
        
        try:
            content = self._call_api(system_prompt, user_prompt)
            result = self._parse_response(content)
            return result.get('is_relevant', False), result.get('reason', ''), result.get('confidence', 0.0)
        except Exception as e:
            print(f"大模型判断出错: {e}")
            # 降级：简单的关键词匹配
            title_desc = (video_info.get('title', '') + ' ' + video_info.get('desc', '')).lower()
            for kw in keywords_str.split(','):
                if kw.strip().lower() in title_desc:
                    return True, "关键词匹配（降级判断）", 0.5
            return False, "关键词不匹配（降级判断）", 0.5
    
    def _call_api(self, system_prompt, user_prompt, timeout=None, max_retries=10):
        """调用大模型API"""
        if timeout is None:
            timeout = int(os.environ.get("SCRAPING_LLM_TIMEOUT", "60"))
        parsed = urlparse(self.base_url)
        host = parsed.netloc
        scheme = parsed.scheme
        path_prefix = parsed.path.rstrip('/')
        endpoint_path = f"{path_prefix}/v1/chat/completions"
        
        payload = json.dumps({
            "model": "gpt-5-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3
        })
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        last_error = None
        for attempt in range(max_retries):
            conn = None
            try:
                if attempt > 0:
                    print(f"重试 {attempt}/{max_retries-1}...")
                    time.sleep(2)  # 重试前等待2秒
                
                if scheme == 'https':
                    conn = http.client.HTTPSConnection(host, timeout=timeout)
                else:
                    conn = http.client.HTTPConnection(host, timeout=timeout)
                
                print(f"正在调用API: {scheme}://{host}{endpoint_path}")
                conn.request("POST", endpoint_path, payload, headers)
                res = conn.getresponse()
                data = res.read()
                response_text = data.decode("utf-8")
                
                print(f"API响应状态: {res.status}")
                
                if res.status != 200:
                    raise Exception(f"API返回错误状态码: {res.status}, 响应: {response_text[:200]}")
                
                response_json = json.loads(response_text)
                
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    content = response_json['choices'][0]['message']['content']
                    print("✅ API调用成功")
                    return content
                else:
                    raise Exception(f"API返回格式错误: {response_text[:200]}")
            
            except Exception as e:
                last_error = e
                print(f"❌ API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            finally:
                if conn:
                    conn.close()
        
        # 所有重试都失败
        raise Exception(f"API调用失败，已重试{max_retries}次。最后错误: {last_error}")
    
    def _parse_response(self, content):
        """解析大模型返回的JSON"""
        # 去除可能的markdown代码块标记
        content = content.strip()
        if content.startswith('```'):
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'```\s*$', '', content)
        
        # 尝试提取JSON对象
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        # 降级返回
        return {"is_relevant": False, "reason": "解析失败", "confidence": 0.0}


def build_fallback_keywords(category, languages, num_keywords):
    """Build deterministic fallback keywords without calling the LLM."""
    config = CATEGORY_CONFIGS[category]
    fallback_map = config.get("fallback_map", {})
    keywords = []
    seen = set()

    for lang in [part.strip() for part in languages.split(",") if part.strip()]:
        for keyword in fallback_map.get(lang, []):
            key = keyword.lower()
            if key in seen:
                continue
            seen.add(key)
            keywords.append(keyword)
            if len(keywords) >= num_keywords:
                return keywords

    for lang_keywords in fallback_map.values():
        for keyword in lang_keywords:
            key = keyword.lower()
            if key in seen:
                continue
            seen.add(key)
            keywords.append(keyword)
            if len(keywords) >= num_keywords:
                return keywords

    return keywords


CATEGORY_KEYWORD_ANCHORS = {
    "stage_performing_arts": ["舞台", "戏剧", "话剧", "歌剧", "音乐剧", "剧场", "剧院", "灯光", "舞美", "导演", "表演"],
    "game_arts": ["游戏", "美术", "视觉", "叙事", "关卡", "CG", "动画", "渲染", "原画", "动作捕捉", "电影化"],
}

CATEGORY_IDENTITY_ANCHORS = {
    "stage_performing_arts": ["舞台", "戏剧", "话剧", "歌剧", "音乐剧", "剧场", "剧院"],
    "game_arts": ["游戏", "CG", "动画", "渲染", "原画", "动作捕捉"],
}

CATEGORY_PRIMARY_ANCHORS = {
    "stage_performing_arts": "舞台",
    "game_arts": "游戏",
}


def _matched_anchor_count(keyword, anchors):
    text = keyword.lower()
    return len({anchor for anchor in anchors if anchor.lower() in text})


def score_search_keyword(keyword, category=None):
    identity_anchors = CATEGORY_IDENTITY_ANCHORS.get(category, [])
    anchors = CATEGORY_KEYWORD_ANCHORS.get(category, [])
    return (
        _matched_anchor_count(keyword, identity_anchors),
        _matched_anchor_count(keyword, anchors),
        len(keyword),
    )


def normalize_search_keyword(keyword, max_length=28, category=None):
    """Compress verbose LLM output into a query string suitable for site search."""
    if not keyword:
        return ""

    raw = str(keyword).strip()
    raw = re.sub(r"（[^）]*）|\([^)]*\)|【[^】]*】|\[[^\]]*\]", "", raw)

    segments = []
    for segment in re.split(r"[：:，,、|｜/]+", raw):
        cleaned = re.sub(r"[《》“”\"'`]", "", segment)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            segments.append(cleaned)

    normalized = ""
    anchors = CATEGORY_KEYWORD_ANCHORS.get(category, [])
    if anchors:
        for segment in segments:
            if any(anchor.lower() in segment.lower() for anchor in anchors):
                normalized = segment
                break

    if not normalized:
        normalized = segments[0] if segments else raw

    normalized = re.sub(r"[《》“”\"'`]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    identity_anchors = CATEGORY_IDENTITY_ANCHORS.get(category, [])
    primary_anchor = CATEGORY_PRIMARY_ANCHORS.get(category)
    if primary_anchor and identity_anchors and not any(anchor.lower() in normalized.lower() for anchor in identity_anchors):
        normalized = f"{primary_anchor} {normalized}".strip()

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


def normalize_search_keywords(keywords, num_keywords, max_length=28, category=None):
    normalized = []
    seen = set()
    for keyword in keywords:
        candidate = normalize_search_keyword(keyword, max_length=max_length, category=category)
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)

    normalized.sort(key=lambda keyword: score_search_keyword(keyword, category=category), reverse=True)
    return normalized[:num_keywords]


NO_AI_RELEVANCE_RULES = {
    "stage_performing_arts": {
        "strong_positive": [
            "舞台", "戏剧", "话剧", "歌剧", "音乐剧", "剧院", "剧场",
            "stage", "theater", "theatre", "opera", "musical",
            "set design", "lighting design",
        ],
        "positive": [
            "舞台", "戏剧", "话剧", "歌剧", "音乐剧", "剧院", "剧场", "表演",
            "导演", "演员", "布景", "灯光", "咏叹调", "stage", "theater",
            "theatre", "opera", "musical", "set design", "lighting design",
        ],
        "analysis": [
            "分析", "解析", "赏析", "评论", "解读", "讲解", "设计", "美学",
            "调度", "体系", "方法", "教程", "essay", "analysis", "breakdown",
            "explained", "critique", "review", "design", "technique",
        ],
        "negative": [
            "游戏", "adofai", "phigros", "maimai", "minecraft", "我的世界",
            "原神", "崩坏", "王者荣耀", "英雄联盟", "通关", "攻略", "实况",
            "直播录像", "谱面", "关卡", "boss", "kpl", "lpl", "电竞",
            "春季赛", "夏季赛", "总决赛",
        ],
    },
    "game_arts": {
        "strong_positive": [
            "游戏美术", "游戏视觉", "视觉叙事", "环境叙事", "关卡设计",
            "cg", "动画", "色彩", "渲染", "像素艺术", "卡通渲染",
            "动作捕捉", "电影化", "art direction", "visual storytelling",
            "level design", "rendering", "animation",
        ],
        "positive": [
            "游戏", "game", "visual", "视觉", "美术", "cg", "动画", "镜头",
            "叙事", "关卡设计", "环境叙事", "色彩", "渲染", "像素艺术",
            "手绘", "卡通渲染", "动作捕捉", "电影化", "art direction",
            "visual storytelling", "level design", "rendering", "animation",
        ],
        "analysis": [
            "分析", "解析", "赏析", "评论", "解读", "讲解", "设计", "美学",
            "风格", "原画", "镜头语言", "视觉引导", "技术", "essay",
            "analysis", "breakdown", "explained", "critique", "review",
        ],
        "negative": [
            "攻略", "实况", "直播", "抽卡", "开箱", "更新公告", "补丁",
            "通关", "全流程", "速通", "赛事", "比赛", "pvp", "配装",
            "build", "mod安装", "下载教程", "推荐", "盘点", "排行",
            "榜单", "风评", "历年最佳", "获奖", "奖项",
        ],
    },
}


def judge_video_relevance_without_ai(video_info, category):
    """Rule-based fallback used when the LLM API is unavailable."""
    rules = NO_AI_RELEVANCE_RULES.get(category)
    if not rules:
        return True, "无类别规则，保守放行", 0.5

    text = " ".join([
        str(video_info.get("title", "")),
        str(video_info.get("desc", "")),
        str(video_info.get("owner", "")),
    ]).lower()

    positive_hits = [term for term in rules["positive"] if term.lower() in text]
    strong_hits = [term for term in rules["strong_positive"] if term.lower() in text]
    analysis_hits = [term for term in rules["analysis"] if term.lower() in text]
    negative_hits = [term for term in rules["negative"] if term.lower() in text]

    if not positive_hits:
        return False, "标题/描述缺少该类别的核心词", 0.2

    if category == "stage_performing_arts" and not strong_hits:
        return False, "缺少舞台/剧场/戏剧/歌剧/音乐剧等强类别词", 0.3

    if category == "stage_performing_arts" and negative_hits:
        return False, f"命中舞台类硬负例词: {', '.join(negative_hits[:3])}", 0.2

    if category == "stage_performing_arts" and not analysis_hits:
        return False, "缺少讲解/解析/设计/评论等分析或设计语境", 0.35

    if negative_hits and not strong_hits:
        return False, f"命中疑似不相关词且缺少强类别词: {', '.join(negative_hits[:3])}", 0.25

    if negative_hits and not analysis_hits:
        return False, f"命中疑似不相关词: {', '.join(negative_hits[:3])}", 0.3

    if analysis_hits:
        confidence = 0.75 if not negative_hits else 0.6
        reason = f"命中类别词: {', '.join(positive_hits[:3])}; 命中分析/设计词: {', '.join(analysis_hits[:3])}"
        return True, reason, confidence

    if len(positive_hits) >= 2 and not negative_hits:
        return True, f"命中多个类别词: {', '.join(positive_hits[:4])}", 0.55

    return False, "类别词较弱且缺少分析/设计语境", 0.35


class BilibiliDownloader:
    def __init__(self, output_dir=str(VIDEO_BILIBILI_ROOT), category="cinematic_arts", min_confidence=0.6):
        """
        初始化B站下载器

        Args:
            output_dir: 视频保存目录
            category: 内容类别
            min_confidence: 最低置信度阈值
        """
        self.category = category
        self.min_confidence = min_confidence
        self.output_dir = Path(output_dir) / category
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 记录文件默认在 project_paths.TRACKING_ROOT 下，也可用环境变量覆写到新结构
        tracking_root = Path(os.environ.get("AUDIOVISUAL_BENCH_TRACKING_ROOT", str(TRACKING_ROOT)))
        record_dir = tracking_root / "bilibili" / category
        record_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_record_file = record_dir / "downloaded_bvids.txt"

        # Migration: copy old tracking files to new location
        if not self.downloaded_record_file.exists():
            import shutil
            for old in [
                record_dir / ".downloaded_bvids.txt",
                Path(output_dir) / "downloaded_bvids.txt",
                Path(output_dir) / ".downloaded_bvids.txt",
                Path(output_dir) / category / "downloaded_bvids.txt",
                Path(output_dir) / category / ".downloaded_bvids.txt",
            ]:
                if old.exists():
                    shutil.copy2(old, self.downloaded_record_file)
                    break
        self.tracked_bvids = self._load_downloaded_records()
        self.downloaded_bvids, self.downloaded_titles = self._load_existing_download_inventory(Path(output_dir))
        self.download_lock = threading.Lock()  # 线程锁保护下载记录
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Origin': 'https://www.bilibili.com',
            'Cookie': 'buvid3=; SESSDATA=; bili_jct=; DedeUserID=; DedeUserID__ckMd5=',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _load_downloaded_records(self):
        """加载已下载的BV号记录"""
        if self.downloaded_record_file.exists():
            try:
                with open(self.downloaded_record_file, 'r', encoding='utf-8') as f:
                    return set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"加载下载记录失败: {e}")
                return set()
        return set()

    def _candidate_scan_roots(self, output_dir_root):
        roots = []
        seen = set()

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

    def _iter_existing_category_dirs(self, root):
        candidates = [
            root / self.category,
            root / "bilibili" / self.category,
            root / "bilibili_videos" / self.category,
        ]
        platform_parents = []
        if root.name in {"bilibili", "bilibili_videos"}:
            platform_parents.append(root)
        for platform_dir_name in ["bilibili", "bilibili_videos"]:
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

        category_dirs = []
        seen = set()
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
            category_dirs.append(candidate)
        return category_dirs

    @staticmethod
    def _safe_title(title):
        return re.sub(r'[\\/:*?"<>|]', '_', title).strip()

    def _load_existing_download_inventory(self, output_dir_root):
        existing_bvids = set()
        existing_titles = set()

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
                        title = metadata.get("title")
                        if source_id:
                            existing_bvids.add(source_id)
                        if title:
                            existing_titles.add(self._safe_title(title))

                for video_path in category_dir.iterdir():
                    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_FILE_EXTENSIONS:
                        continue
                    stem = video_path.stem.strip()
                    if "__" in stem:
                        bvid, title = stem.split("__", 1)
                        bvid = bvid.strip()
                        title = title.strip()
                        if bvid.startswith("BV"):
                            existing_bvids.add(bvid)
                        if title:
                            existing_titles.add(self._safe_title(title))
                    elif stem:
                        existing_titles.add(self._safe_title(stem))

        return existing_bvids, existing_titles

    def _save_downloaded_record(self, bvid, title=None):
        """保存已下载的BV号"""
        try:
            with self.download_lock:
                self.downloaded_bvids.add(bvid)
                if title:
                    self.downloaded_titles.add(self._safe_title(title))
                with open(self.downloaded_record_file, 'a', encoding='utf-8') as f:
                    f.write(f"{bvid}\n")
        except Exception as e:
            print(f"保存下载记录失败: {e}")

    def is_downloaded(self, bvid, title=None):
        """检查视频是否已下载"""
        with self.download_lock:
            if bvid in self.downloaded_bvids:
                return True
            if title and self._safe_title(title) in self.downloaded_titles:
                return True
            return False
    
    def extract_bvid(self, url_or_bvid):
        """
        从URL或直接BV号中提取BV号
        
        Args:
            url_or_bvid: B站视频URL或BV号
            
        Returns:
            BV号字符串
        """
        # 如果已经是BV号格式
        if url_or_bvid.startswith('BV'):
            return url_or_bvid
        
        # 从URL中提取BV号
        patterns = [
            r'BV[a-zA-Z0-9]+',
            r'/video/(BV[a-zA-Z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_bvid)
            if match:
                if 'BV' in match.group(0):
                    return match.group(0)
                return match.group(1)
        
        raise ValueError(f"无法从 '{url_or_bvid}' 中提取BV号")
    
    def get_video_info(self, bvid):
        """
        获取视频信息
        
        Args:
            bvid: BV号
            
        Returns:
            视频信息字典
        """
        api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        
        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != 0:
                raise Exception(f"获取视频信息失败: {data.get('message', '未知错误')}")
            
            video_data = data['data']
            
            info = {
                'bvid': video_data['bvid'],
                'aid': video_data['aid'],
                'title': video_data['title'],
                'desc': video_data['desc'],
                'duration': video_data['duration'],
                'owner': video_data['owner']['name'],
                'cid': video_data['cid'],
                'pic': video_data['pic'],
            }
            
            print(f"\n视频信息:")
            print(f"标题: {info['title']}")
            print(f"UP主: {info['owner']}")
            print(f"时长: {info['duration']}秒")
            print(f"BV号: {info['bvid']}")
            
            return info
            
        except Exception as e:
            print(f"获取视频信息出错: {e}")
            raise
    
    def search_videos(self, keyword, page=1, page_size=20):
        """
        搜索B站视频
        
        Args:
            keyword: 搜索关键词
            page: 页码（从1开始）
            page_size: 每页结果数量
            
        Returns:
            视频信息列表
        """
        params = {
            'search_type': 'video',
            'keyword': keyword,
            'page': page,
            'page_size': page_size,
            'order': 'totalrank',  # 综合排序
        }
        
        api_urls = [
            "https://api.bilibili.com/x/web-interface/wbi/search/type",
            "https://api.bilibili.com/x/web-interface/search/type",
        ]

        last_error = None
        for api_url in api_urls:
            try:
                response = self.session.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()

                if data['code'] != 0:
                    raise Exception(f"搜索失败: {data.get('message', '未知错误')}")

                results = []
                if 'result' in data['data'] and data['data']['result']:
                    for item in data['data']['result']:
                        # 清理HTML标签
                        title = re.sub(r'<[^>]+>', '', item.get('title', ''))
                        desc = re.sub(r'<[^>]+>', '', item.get('description', ''))

                        video_info = {
                            'bvid': item.get('bvid', ''),
                            'title': title,
                            'desc': desc,
                            'owner': item.get('author', ''),
                            'duration': item.get('duration', ''),
                            'play': item.get('play', 0),
                            'pic': item.get('pic', ''),
                            'arcurl': item.get('arcurl', ''),
                        }
                        results.append(video_info)

                return results

            except Exception as e:
                last_error = e
                continue

        print(f"搜索视频出错: {last_error}")
        return []
    
    def get_download_url(self, bvid, cid, quality=80):
        """
        获取视频下载链接
        
        Args:
            bvid: BV号
            cid: 视频CID
            quality: 视频质量 (80=1080P, 64=720P, 32=480P, 16=360P)
            
        Returns:
            视频和音频的下载链接
        """
        api_url = f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&qn={quality}&fnval=16"
        
        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != 0:
                raise Exception(f"获取下载链接失败: {data.get('message', '未知错误')}")
            
            # 获取视频和音频流
            dash = data['data']['dash']
            video_url = dash['video'][0]['baseUrl']  # 选择第一个视频流
            audio_url = dash['audio'][0]['baseUrl']  # 选择第一个音频流
            
            return video_url, audio_url
            
        except Exception as e:
            print(f"获取下载链接出错: {e}")
            raise
    
    def download_file(self, url, output_path):
        """
        下载文件
        
        Args:
            url: 下载链接
            output_path: 输出路径
        """
        try:
            print(f"开始下载: {output_path.name}")
            
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n下载完成: {output_path}")
            
        except Exception as e:
            print(f"\n下载失败: {e}")
            raise
    
    def merge_video_audio(self, video_path, audio_path, output_path):
        """
        合并视频和音频文件（需要ffmpeg）
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出文件路径
        """
        try:
            import subprocess
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                str(output_path),
                '-y'  # 覆盖已存在的文件
            ]
            
            print(f"\n合并视频和音频...")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"合并完成: {output_path}")
            
            # 删除临时文件
            video_path.unlink()
            audio_path.unlink()
            print("已删除临时文件")
            
        except FileNotFoundError:
            print("错误: 未找到ffmpeg。请先安装ffmpeg: brew install ffmpeg")
            print(f"视频文件: {video_path}")
            print(f"音频文件: {audio_path}")
        except Exception as e:
            print(f"合并失败: {e}")
    
    def download_video(self, url_or_bvid, quality=80):
        """
        下载B站视频
        
        Args:
            url_or_bvid: B站视频URL或BV号
            quality: 视频质量 (80=1080P, 64=720P, 32=480P, 16=360P)
        """
        try:
            # 提取BV号
            bvid = self.extract_bvid(url_or_bvid)
            print(f"解析到BV号: {bvid}")
            
            # 获取视频信息
            info = self.get_video_info(bvid)
            
            # 获取下载链接
            print("\n获取下载链接...")
            video_url, audio_url = self.get_download_url(bvid, info['cid'], quality)
            
            # 清理文件名
            safe_title = self._safe_title(info['title'])
            base_name = f"{bvid}__{safe_title}"

            # 下载视频和音频
            video_temp = self.output_dir / f"{base_name}_video.m4s"
            audio_temp = self.output_dir / f"{base_name}_audio.m4s"
            output_file = self.output_dir / f"{base_name}.mp4"
            
            print("\n下载视频流...")
            self.download_file(video_url, video_temp)
            
            print("\n下载音频流...")
            self.download_file(audio_url, audio_temp)
            
            # 合并视频和音频
            self.merge_video_audio(video_temp, audio_temp, output_file)

            # 记录已下载
            self._save_downloaded_record(bvid, info['title'])
            
            print(f"\n✅ 视频下载完成!")
            print(f"保存位置: {output_file}")
            
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            raise
    
    def _download_video_task(self, video, keywords, llm_client, quality, task_id, total_tasks):
        """
        单个视频下载任务（用于多线程）
        
        Returns:
            (success: bool, bvid: str, title: str)
        """
        try:
            print(f"\n{'='*60}")
            print(f"[任务 {task_id}/{total_tasks}] 视频: {video['title']}")
            print(f"UP主: {video['owner']}")
            print(f"BV号: {video['bvid']}")

            # 检查是否已下载
            if self.is_downloaded(video['bvid'], video['title']):
                print("⏭️  已下载过，跳过")
                return False, video['bvid'], video['title']
            
            # AI判断相关性
            if llm_client:
                print("\n使用AI判断视频相关性...")
                is_relevant, reason, confidence = llm_client.judge_video_relevance(video, keywords, category=self.category)
                print(f"判断结果: {'✅ 相关' if is_relevant else '❌ 不相关'}")
                print(f"理由: {reason}")
                print(f"置信度: {confidence:.2f}")
                
                if not is_relevant or confidence < self.min_confidence:
                    print("跳过此视频")
                    return False, video['bvid'], video['title']
            else:
                print("\n使用规则判断视频相关性...")
                is_relevant, reason, confidence = judge_video_relevance_without_ai(video, self.category)
                print(f"判断结果: {'✅ 相关' if is_relevant else '❌ 不相关'}")
                print(f"理由: {reason}")
                print(f"置信度: {confidence:.2f}")

                if not is_relevant:
                    print("跳过此视频")
                    return False, video['bvid'], video['title']
            
            # 下载视频
            print(f"\n开始下载...")
            self.download_video(video['bvid'], quality=quality)
            print(f"✅ 下载完成: {video['title']}")
            return True, video['bvid'], video['title']
            
        except Exception as e:
            print(f"❌ 任务失败 [{video['title']}]: {e}")
            return False, video['bvid'], video['title']
    
    def search_and_download(self, keywords, max_videos=5, llm_client=None, quality=80, max_workers=3):
        """持续搜索关键词并下载视频，直到完成目标数量或无片可下"""
        if isinstance(keywords, str):
            keywords_list = [keywords]
        else:
            keywords_list = list(keywords)
        
        keywords_list = normalize_search_keywords(keywords_list, len(keywords_list), category=self.category)
        if not keywords_list:
            print("\n❌ 关键词列表为空，无法开始下载")
            return
        
        keyword_pages = {kw: 1 for kw in keywords_list}
        exhausted_keywords = set()
        seen_bvids = set()
        downloaded_count = 0
        download_target_met = False
        search_round = 1
        
        while downloaded_count < max_videos and len(exhausted_keywords) < len(keywords_list):
            print(f"\n{'='*60}")
            print(f"第 {search_round} 轮搜索 (目标: {downloaded_count}/{max_videos})")
            print(f"{'='*60}")
            search_round += 1
            
            for keyword in keywords_list:
                if downloaded_count >= max_videos:
                    download_target_met = True
                    break
                if keyword in exhausted_keywords:
                    continue
                
                page = keyword_pages[keyword]
                print(f"\n{'-'*60}")
                print(f"搜索关键词: {keyword} (第 {page} 页)")
                print(f"{'-'*60}")
                videos = self.search_videos(keyword, page=page, page_size=30)
                
                if not videos:
                    print("该关键词没有更多搜索结果，标记为耗尽")
                    exhausted_keywords.add(keyword)
                    continue
                
                keyword_pages[keyword] += 1
                new_videos = []
                for video in videos:
                    bvid = video.get('bvid')
                    if not bvid:
                        continue
                    if bvid not in seen_bvids:
                        seen_bvids.add(bvid)
                        new_videos.append(video)
                
                if not new_videos:
                    print("本页均为已处理的BV号，继续翻页...")
                    continue
                
                print(f"发现 {len(new_videos)} 个全新视频，准备下载")
                
                for batch_start in range(0, len(new_videos), max_workers):
                    if downloaded_count >= max_videos:
                        download_target_met = True
                        break

                    batch = new_videos[batch_start:batch_start + max_workers]
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_video = {}
                        for offset, video in enumerate(batch, 1):
                            task_idx = batch_start + offset
                            future = executor.submit(
                                self._download_video_task,
                                video, keywords_list, llm_client, quality, task_idx, len(new_videos)
                            )
                            future_to_video[future] = video

                        for future in as_completed(future_to_video):
                            try:
                                success, bvid, title = future.result()
                                if success:
                                    downloaded_count += 1
                                    print(f"\n📊 进度: 已下载 {downloaded_count}/{max_videos} 个视频")
                                    if downloaded_count >= max_videos:
                                        download_target_met = True
                            except Exception as e:
                                print(f"\n⚠️  任务异常: {e}")

                    if download_target_met:
                        break
                
                if download_target_met:
                    break
            
        print(f"\n{'='*60}")
        if downloaded_count >= max_videos:
            print(f"🎉 达成目标! 共下载 {downloaded_count} 个视频")
        else:
            print(f"⚠️ 搜索耗尽，仍有 {max_videos - downloaded_count} 个视频未完成")
            if exhausted_keywords:
                remaining = set(keywords_list) - exhausted_keywords
                print(f"已耗尽关键词数量: {len(exhausted_keywords)}，剩余可用: {len(remaining)}")
        print(f"{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Bilibili 智能视频下载器")
    parser.add_argument("--category", choices=list(CATEGORY_CONFIGS.keys()), default="cinematic_arts")
    parser.add_argument("--max", type=int, default=500)
    parser.add_argument("--quality", type=int, default=64)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--topic", type=str, default="电影拉片解说分析，重点关注电影技法运用：蒙太奇、镜头语言、场面调度、希区柯克悬念、长镜头、构图、光影、色彩、视听语言、剪辑手法等电影艺术技巧")
    parser.add_argument("--output", type=str, default=str(VIDEO_BILIBILI_ROOT))
    parser.add_argument("--api-key", type=str, default=os.environ.get("BILIBILI_API_KEY") or os.environ.get("API_KEY"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("BILIBILI_BASE_URL") or (f"https://{os.environ['API_HOST']}" if os.environ.get("API_HOST") else "https://jeniya.top"))
    parser.add_argument("--languages", type=str, default="zh-CN")
    parser.add_argument("--num-keywords", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--no-ai", action="store_true", help="不调用LLM，使用预设关键词并跳过AI相关性判断")
    args = parser.parse_args()
    if not args.api_key and not args.no_ai:
        parser.error("--api-key or BILIBILI_API_KEY is required")

    print("=" * 50)
    print("Bilibili 智能视频下载器")
    print("=" * 50)
    print(f"\n📂 类别: {args.category}")
    print(f"📹 默认设置: 画质{args.quality}，最多下载{args.max}个视频")
    print("🤖 AI智能: 自动生成关键词并判断视频相关性" if not args.no_ai else "🤖 AI智能: 已关闭，使用预设关键词直接抓取")
    print("♻️  智能去重: 自动跳过已下载视频")
    print(f"⚡ 多线程: {args.workers}个并发线程加速下载\n")

    if args.no_ai:
        llm_client = None
        print(f"\n使用预设关键词爬取主题『{args.topic}』...")
        keywords = build_fallback_keywords(args.category, args.languages, args.num_keywords)
    else:
        # 创建LLM客户端
        print("正在初始化AI...")
        llm_client = LLMClient(api_key=args.api_key, base_url=args.base_url)
        print("✅ AI已就绪")

        # 使用AI生成搜索关键词
        print(f"\n正在为主题『{args.topic}』生成搜索关键词...")
        keywords = llm_client.generate_search_keywords(args.topic, num_keywords=args.num_keywords, language=args.languages, category=args.category)
    print(f"✅ 生成关键词: {', '.join(keywords)}\n")

    # 创建下载器实例
    downloader = BilibiliDownloader(output_dir=args.output, category=args.category, min_confidence=args.min_confidence)

    # 开始搜索和下载
    downloader.search_and_download(
        keywords,
        max_videos=args.max,
        llm_client=llm_client,
        quality=args.quality,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
