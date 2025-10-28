# -*- coding: utf-8 -*-
import argparse
import os
import random
import json
import pandas as pd
import openai
from api_key import OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY",OPENAI_API_KEY)
import torch
from preprocess import load_skills_data, preprocess_dataset
from run import run_openai
from utils.chat_utils import safe_chat
from evaluate_src import eval
from demo_retrieval import embed_demo_dataset
import shutil
from classification_prompt import CLASSIFICATION_PROMPT
random.seed(1234)
#新增代码
from utils.io_utils import read_json, write_json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
_requests_lock = threading.Lock()
_last_ts = [0.0]

_ALLOWED_BASE_SPLITS = {'train', 'test', 'dev', 'val', 'valid'}

def _infer_base_split(split_name: str) -> str:
    """
    从 'dev.p000' 推断基础 split -> 'dev'；若不在白名单，则原样返回。
    """
    if not isinstance(split_name, str) or not split_name:
        return 'test'
    base = split_name.split('.')[0]
    return base if base in _ALLOWED_BASE_SPLITS else split_name

def _resolve_split_json(processed_dir: str, split: str) -> str:
    """
    在 processed_dir 下按以下顺序查找：
      1) <processed_dir>/<split>.json                (e.g., dev.p000.json)
      2) <processed_dir>/<base>/<split>.json         (e.g., dev/dev.p000.json)
      3) <processed_dir>/<base>.json                 (e.g., dev.json)
    若都不存在，返回 1) 的路径作为“期望路径”，由调用处决定报错或提示。
    """
    base = _infer_base_split(split)
    candidates = [
        os.path.join(processed_dir, f"{split}.json"),
        os.path.join(processed_dir, base, f"{split}.json"),
        os.path.join(processed_dir, f"{base}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 返回首选路径，方便上游统一提示
    return candidates[0]


_ASCII_RUN = re.compile(r'[A-Za-z0-9]+')

def _tokenize_zh(text: str):
    """
    与预处理里用的简易分词保持一致：中文逐字，ASCII 连续段合并。
    让 run_openai 的 BIO 长度对齐。
    """
    toks = []
    if not isinstance(text, str):
        text = str(text or "")
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _ASCII_RUN.match(ch):
            m = _ASCII_RUN.match(text, i)
            seg = m.group(0)
            toks.append(seg)
            i += len(seg)
        else:
            toks.append(ch)
            i += 1
    return toks

def _load_paragraph_jsonl(paragraph_path: str,
                          user_text_keys: str = None,
                          user_id_keys: str = None,
                          join_lists: bool = False):
    """
    兼容 JSONL / JSON数组；更强健的文本提取与回退策略：
    1) 优先使用用户传入的文本键名（逗号分隔）。
    2) 再使用内置候选键名。
    3) 若仍未命中：从该条记录所有 str 字段里选取“最长”的作为段落；
       若存在 list[str] 且 join_lists=True，则以 '\\n' 拼接。
    ID 提取同理：先用户键 → 内置候选 → 自动生成。
    """
    import io

    def _read_records(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return []
        # JSON 数组
        if raw.startswith("[") and raw.endswith("]"):
            try:
                arr = json.loads(raw)
                return [x for x in arr if isinstance(x, dict)]
            except Exception:
                pass
        # JSONL
        recs = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    recs.append(obj)
            except Exception:
                continue
        return recs

    # 解析候选键
    text_keys = []
    if user_text_keys:
        text_keys.extend([k.strip() for k in user_text_keys.split(",") if k.strip()])

    builtin_text_keys = [
        "paragraph", "text", "sentence", "content",
        "raw_text", "cleaned_text", "full_text", "doc", "jd", "JD",
        "文本", "正文", "原文", "招聘文本", "职位描述", "工作描述", "岗位描述", "岗位职责", "职责"
    ]
    # 去重
    for k in builtin_text_keys:
        if k not in text_keys:
            text_keys.append(k)

    id_keys = []
    if user_id_keys:
        id_keys.extend([k.strip() for k in user_id_keys.split(",") if k.strip()])
    builtin_gid_keys = ["global_id", "Global_ID", "Gloobal_ID", "Globa_ID", "globalId"]
    builtin_id_keys  = ["id", "ID", "doc_id", "uid", "uniq_id"]
    for k in builtin_gid_keys + builtin_id_keys:
        if k not in id_keys:
            id_keys.append(k)

    recs = _read_records(paragraph_path)
    if not recs:
        print(f"[paragraph] 文件为空或解析失败：{paragraph_path}")
        return []

    # 打印一条样本的全部键，方便排查字段名
    sample_keys = list(recs[0].keys())
    print(f"[paragraph] example keys of first record: {sample_keys}")

    rows, miss_cnt = [], 0
    for idx, rec in enumerate(recs, 1):
        para = None

        # 1) 直接在候选键里找字符串或列表
        for k in text_keys:
            if k in rec and rec[k] is not None:
                v = rec[k]
                # 字符串
                if isinstance(v, str) and v.strip():
                    para = v.strip()
                    break
                # 列表[str]
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    if join_lists and len(v) > 0:
                        para = "\n".join([x.strip() for x in v if x and x.strip()])
                        if para.strip():
                            break
                    # 不拼接则继续看其它键

        # 2) 仍未命中：回退到“最长字符串字段”或“可拼接的字符串列表”
        if not para:
            longest_text = ""
            list_text = ""
            if join_lists:
                # 找所有列表[str]，拼接后选最长
                for k, v in rec.items():
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        joined = "\n".join([x.strip() for x in v if x and x.strip()])
                        if len(joined) > len(list_text):
                            list_text = joined
            for k, v in rec.items():
                if isinstance(v, str) and v.strip():
                    if len(v) > len(longest_text):
                        longest_text = v.strip()
            # 优先用列表拼接结果（更像段落），否则用最长字符串字段
            para = list_text if list_text.strip() else longest_text

        if not para:
            miss_cnt += 1
            continue

        # 3) 提取 id/global_id
        gid, rid = None, None
        for k in id_keys:
            if k in rec and str(rec[k]).strip():
                # 第一个命中的当作 gid，再找下一命中当 rid；或者直接都设为相同
                if gid is None:
                    gid = str(rec[k]).strip()
                elif rid is None:
                    rid = str(rec[k]).strip()
                    break
        if gid is None:
            gid = f"pg-{idx}"
        if rid is None:
            rid = f"{gid}-p{idx:04d}"

        # 4) 额外元数据尝试
        src = rec.get("source_domain") or rec.get("domain") or ""
        job = rec.get("job_title") or rec.get("职位") or rec.get("岗位") or rec.get("职位名称") or ""

        rows.append({
            "id": rid,
            "global_id": gid,
            "sent_id": 0,
            "sentence": para,
            "tokens": list(para),     # 简单占位；后续不会用到 tokens 对段落
            "skill_spans": [],
            "tags_skill": [],
            "tags_skill_clean": [],
            "sentence_with_tags": para,
            "source_domain": src,
            "job_title": job,
        })

    print(f"[paragraph] scanned={len(recs)}, ok={len(rows)}, skipped(no-text)={miss_cnt}")
    return rows



def _anchor_sentence_from_sample(sample: dict) -> str:
    s = sample.get('sentence_with_tags') or sample.get('sentence') or ""
    if '@@' in s and '##' in s:
        return s.replace('@@', '[SKILL] ').replace('##', ' [/SKILL]')
    tokens = sample.get('tokens')
    tags = sample.get('tags_skill_clean') or sample.get('tags_skill') or sample.get('labels')
    if not (tokens and tags) or len(tokens) != len(tags):
        return sample.get('sentence', ' '.join(tokens or []))
    out, i, n = [], 0, len(tokens)
    while i < n:
        if tags[i] == 'B':
            j = i
            while j+1 < n and tags[j+1] == 'I':
                j += 1
            span = " ".join(tokens[i:j+1])
            out.append(f"[SKILL] {span} [/SKILL]")
            i = j + 1
        else:
            out.append(tokens[i]); i += 1
    return " ".join(out)

def _norm_cls_label(s: str) -> str:
    s = (s or "").strip().lower().replace("\u202f"," ").replace("\xa0"," ").replace("  "," ")
    if "non" in s and "skill" in s:
        return "Non-Skill"
    if "skill" in s:
        return "Skill"
    return "Non-Skill"


def download(args, split):
    # ESCO 数据集为本地预置，无需下载  (跳过 download 阶段)
    if args.dataset_name.lower() == 'esco':
        print(f'Using local ESCO dataset, skip download. Path: {args.raw_data_dir}')
        return
    dataset = load_skills_data(args.dataset_name, split)
    dataset.to_json(
        args.raw_data_dir + '/' + split + '.json',
        orient='records',
        indent=4,
        force_ascii=False
    )
    print(f'Saved {args.dataset_name} dataset to {args.raw_data_dir}, with {len(dataset)} examples.')



def parse_args():

    parser = argparse.ArgumentParser(description="Skill-Extraction runner / evaluator")
    # 是否强制重建 .pt（默认不重建，复用）
    parser.add_argument('--force_reembed', action='store_true',
                        help='Force regenerate train/test embedding .pt files even if they already exist.')
    parser.add_argument("--concurrency", type=int, default=4,
                        help="并发请求数（openai 后端生效），建议 2~6")
    parser.add_argument("--request_interval", type=float, default=0.25,
                        help="相邻请求的最小时间间隔（秒），用于节流，示例 0.25=每秒最多4个新请求")

    parser.add_argument('--knn_en', action='store_true',
                        help='Use Skill-Entity Embedding for kNN demo retrieval (anchor spans in text).')
    parser.add_argument('--rag_en', action='store_true',
                        help='Enable entity-level retrieval for checker (per-span ESCO defs).')
    parser.add_argument('--en_mode', choices=['anchor', 'phrase', 'mix'], default='anchor',
                        help='Entity embedding mode: anchor[SKILL]span[/SKILL], phrase-only, or mix.')
    parser.add_argument('--en_mix_alpha', type=float, default=0.7,
                        help='When --en_mode=mix, similarity = alpha*anchor + (1-alpha)*phrase.')

    # parser.add_argument('--split', type=str, default='test',
    #                     choices=['train', 'test', 'dev', 'val', 'valid'])
    parser.add_argument('--split', type=str, default='test',
                        help="数据分片名；支持 train/dev/test 以及 dev.p000 这类分片名")
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--save_jsonl', type=str, default=None)
    parser.add_argument('--save_stem', type=str, default=None)

    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='采样温度；仅当 do_sample=True 时生效。本地 pipeline 会据此切换采样/贪心。'
    )
    parser.add_argument(
        '--top_p', type=float, default=1.0,
        help='nucleus sampling；仅当 do_sample=True 时生效。'
    )

    parser.add_argument('--exp_id', type=str, default=None, help='实验编号，如 B1/B2/O1 等')
    parser.add_argument('--cluster', type=str, default=None, help='聚类方法标签，如 kmeans/lda/agg 等')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='跳过 preprocess_dataset 阶段（与 --force_preprocess 互斥优先被覆盖）')
    parser.add_argument('--force_preprocess', action='store_true',
                        help='无论是否已存在 processed 文件都强制重建')
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="底层全量模型名称或路径（仅 --backend local 时需要）",
    )
    parser.add_argument(
        "--cls_model",
        type=str,
        default=None,
        help="本地分类模型 LoRA adapter 路径；优先于 --local_model 使用",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="先对每条句子做 Skill vs Non‑Skill 分类，再把分类结果注入到后续的标注 prompt 里"
    )
    parser.add_argument(
        '--cls_threshold', type=float, default=0.5,
            help = "本地模型分类时判定为“Skill”的最低置信度阈值（0–1），越低 Recall 越高"
          )
    parser.add_argument(
        '--force_reclassify',
        action='store_true',
        help='即使存在 classification 结果文件，也强制重新跑 classification 阶段'
    )
     # ---------- 1. 数据集 ----------
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='一次性并行推理的样本数；1 表示逐条调用'
    )
    parser.add_argument(
        '--dataset_name', default='None',
        help='Dataset name to use. Options: green, skillspan, fijo, sayfullina, kompetencer, esco,chinese_skillspan"'
    )
    parser.add_argument('--raw_data_dir', type=str, default='data/annotated/raw/')
    parser.add_argument('--processed_data_dir', type=str, default=None)
    # ---------- 2. Prompt ----------
    #parser.add_argument('--prompt_type', type=str, default='ner')
    # parser.add_argument('--prompt_type', type=str, default='ner', choices=['ner', 'extract', 'ner_cot', 'extract_cot'],help='* _cot 结尾将自动在 prompt 中加入 CoT reasoning 指令')
    parser.add_argument(
        '--prompt_module',
        type=str,
        default='prompt_template',
        choices=['prompt_template', 'prompt_template_rag','prompt_template_ctx','prompt_template_dualcot','prompt_template_rag_old',""],
        help='which prompt module to use'
    )

    parser.add_argument(
        '--prompt_type',
        type=str,
        default='ner',
        choices=['ner', 'ner_old', 'extract', 'ner_cot', 'extract_cot', 'ner_cot2'],
        help='prompt type / task mode'
    )
    parser.add_argument("--cot2_output",
                        default="mixed", choices=["mixed", "bio_only"],
                        help="For ner_cot2: whether to return both ANNOTATED_SENTENCE and BIO (mixed) or only BIO line.")
    parser.add_argument(
        "--prompt_style", default=None,
        help="模板风格：generic / fijo …；留空=使用数据集专用模板"
    )
    parser.add_argument(
        "--no_specific", action="store_true",
        help="【已废弃】同 --prompt_style generic，未来将移除",
    )

  #  parser.add_argument('--model', type=str, default='gpt-3.5-turbo')

    # ---------- 3. 运行开关的参数 ----------
    # run parameters
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--rag', action='store_true', help='是否开启RAG检索增强')
    parser.add_argument(
        '--rag_topk',
        type=int,
        default=3,
        help='RAG 检索时每条句子检索的 top-k 数量'
    )
    parser.add_argument(
        '--rag_topk_train',
        type=int,
        default=None,
        help='RAG 检索训练语料时的 top-k 数量，若未指定则等于 --rag_topk'
    )
    parser.add_argument(
        '--rag_topk_kb',
        type=int,
        default=None,
        help='RAG 检索背景／定义语料时的 top-k 数量，若未指定则等于 --rag_topk'
    )
    parser.add_argument(
        "--rag_threshold",
        type=float,
        default=0.6,
        help="最低余弦相似度阈值，小于该值视为无相关项"
    )
    parser.add_argument(
    "--rag_corpus", type=str,
    default="data/rag_skills.json",
    help="RAG 知识库文件（JSON 或 JSONL）路径"
)
    parser.add_argument(
    "--rag_emb", type=str,
    default="data/rag_skills.npy",
    help="RAG 知识库 embeddings (.npy) 路径")
    parser.add_argument('--rag_general_corpus',type=str, default=None,
                        help='通用背景知识库（职业定义），例如 ESCO definitions')
    parser.add_argument("--rag_def_corpus", type=str, default=None, help = "技能定义库（ESCO 技能短定义），jsonl 格式")
    parser.add_argument('--knn',action='store_true',help='是否启用 KNN 检索增强（不需要额外参数）')
    parser.add_argument('--process', action='store_true')
    parser.add_argument('--start_from_saved', action='store_true', help='Start from saved results instead of running inference again.')
    parser.add_argument('--exclude_empty', action='store_true', help='Exclude examples that have no skills in them.')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--positive_only', action='store_true', help='whether to include only positive samples from the dataset')
    parser.add_argument('--sample', type=int, default=0, help='number of samples to perform inference on, for debugging.')
    parser.add_argument('--exclude_failed', action='store_true', help='whether to exclude previous failed attempt')
    parser.add_argument('--verbose', action='store_true',##
                        help='打印模型原始输出及解析后的技能列表')

    parser.add_argument('--debug_raw', action='store_true',
                        help='逐条打印 GPT 返回的原始文本以及解析出的技能列表')

    # ---------- 4. RAG参数设置 ----------
    parser.add_argument('--small_train', type=int, default=None, help='仅用前N条训练样本做Smoke Test')
    parser.add_argument('--small_eval', type=int, default=None, help='仅用前N条验证样本做Smoke Test')
    # ---------- backend ----------
    parser.add_argument('--backend', choices=['openai', 'local'], default='openai')
    # main.py → parse_args() 里，和 --backend / --local_model / --base_model 同一区域
    # —— 自动匹配 LoRA 基座 & 严格校验 开关 ——
    parser.set_defaults(auto_match_base=True)  # 默认开启自动匹配
    parser.add_argument(
        '--auto_match_base', dest='auto_match_base', action='store_true',
        help='当 --local_model 是 LoRA 目录且未显式提供正确的 --base_model 时，自动读取适配器记录的基座并切换。'
    )
    parser.add_argument(
        '--no_auto_match_base', dest='auto_match_base', action='store_false',
        help='关闭自动匹配；遇到不匹配会按 strict 设置处理或直接报错。'
    )
    parser.add_argument(
        '--strict_base_match', action='store_true',
        help='发现 LoRA 记录的 base 与 --base_model 不一致时立刻报错退出（不自动切换）。'
    )
  #  parser.add_argument('--local_model', default='Qwen/Qwen1.5-1.8B-Chat')
    # 二选一：model OR local_model
    # ---------- 5.模型选择 ----------
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--model',
                       help='云端大模型名称，如 gpt-3.5-turbo、gpt-4o、claude-3-opus …')
    group.add_argument('--local_model',
                       help='本地模型 repo id 或文件夹，如 Qwen/Qwen2.5-14B-Instruct')
    parser.add_argument("--api_base", type=str, default=None)
    # parser.add_argument("--rag_backend", choices=["rag_local", "default", "rag_prompt"], default="default")
    parser.add_argument(
        "--rag_backend",
        choices=["rag_local", "default", "rag_prompt"],
        default="default",
        help="选择 RAG 后端：rag_local | default | rag_prompt"
    )
    # ---------- Checking stage ----------
    parser.add_argument('--check', action='store_true',
                        help='Enable post-generation checking & minimal fixes.')
    parser.add_argument('--check_precision_mode', default='balanced',
                        choices=['high','balanced'],
                        help='Checker precision mode: high(更保守,抬precision) / balanced(折中).')
    parser.add_argument('--check_model', type=str, default=None,
                        help='(可选) 指定用于checking的模型，默认复用 --model / --local_model')
    parser.add_argument(
        '--format_retry', type=int, default=2,
        help='格式/长度不合法时对主模型的小循环纠错重试次数（A1 设为 0）'
    )
    parser.add_argument('--retry', type=int, default=10, help='API 调用最大重试次数（默认 10）')

    parser.add_argument(
        '--cls_filter_mode',
        type=str,
        default='none',
        choices=['none', 'soft', 'hard'],
        help='分类为 Non-Skill 时的处理策略：none=仅提示；soft=快路径输出全O并可走check；hard=跳过后续，直接全O不check'
    )
    # 段落模式开关与输入文件
    parser.add_argument("--unit", choices=["sent", "paragraph"], default="sent",
                    help="推理单元：sent=按句（默认），paragraph=按整段文本")
    parser.add_argument("--paragraph_path", default=None,
                    help="段落模式输入文件（paragraph.jsonl 的绝对路径）")
    parser.add_argument("--paragraph_text_keys", default=None,
                        help="段落文本字段名（多个用逗号分隔）。若不提供，将使用内置候选+自动回退（最长字符串 / 拼接字符串列表）。")
    parser.add_argument("--paragraph_id_keys", default=None,
                        help="ID/Global_ID 字段名候选（多个用逗号分隔），优先用于定位 id/global_id。")
    parser.add_argument("--paragraph_join_lists", action="store_true",
                        help="当文本字段是列表[str]时，是否将其用换行拼接成一个段落文本。")
    # 段落上下文文件（给句子提供背景，不改变推理单元）
    parser.add_argument("--paragraph_context_path", default=None,
                        help="可选：为句子级推理提供段落级上下文（paragraph.jsonl 路径）。与 --unit sent 搭配使用")
    # === Context Assist (最小化) ===
    parser.add_argument("--ctx_assist", action="store_true",
                        help="开启情境辅助：在提示词中注入 标题 + 上下句 作为参考，但只标注当前句。")
    parser.add_argument("--ctx_window", type=int, default=1,
                        help="上下文窗口大小（1=上一句/下一句；2=上两句/下两句；…）")

    args = parser.parse_args()
    #   # else:
        #     args.model_name = os.path.basename(args.base_model.rstrip("/"))保留用户的显式选择；仅当需要时做合理推断
    # 仅当本地推理时，才要求提供 local_model 或 base_model
    # 记录原始 split 与基础 split
    args.split_orig = args.split
    args.split_base = _infer_base_split(args.split)

    if args.backend == "local":
        if not (args.local_model or args.base_model):
            parser.error("--backend local 需要提供 --local_model 或 --base_model 其中之一。")
        src = args.local_model or args.base_model
        args.model_name = os.path.basename(src.rstrip("/"))

    else:
        # 云端推理
        args.backend = "openai"
        args.model_name = args.model or "gpt-3.5-turbo"

    args.out_dir = os.path.join("output", args.dataset_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 保存路径优先级（尊重用户传入），并保证四件套成对存在 ---
    user_save_stem = args.save_stem  # 可能是 None 或用户传入
    user_save_path = args.save_path
    user_save_jsonl = args.save_jsonl

    if user_save_stem is None and user_save_path is None and user_save_jsonl is None:
        # 用户没给任何，按 exp_tag 生成默认
        if args.rag_topk_train is None:
            args.rag_topk_train = args.rag_topk
        if args.rag_topk_kb is None:
            args.rag_topk_kb = args.rag_topk
        exp_tag = _build_exp_tag(args)
        stem = os.path.join(args.out_dir, exp_tag)
        args.save_stem = stem
        args.save_path = f"{stem}.json"
        args.save_jsonl = f"{stem}.jsonl"
        args.error_path = f"{stem}.error.jsonl"
    else:
        # 用户给了其中至少一个，按优先级派生其余（stem > path > jsonl）
        if user_save_stem:
            stem = user_save_stem
            args.save_stem = stem
            args.save_path = user_save_path or f"{stem}.json"
            args.save_jsonl = user_save_jsonl or f"{stem}.jsonl"
            args.error_path = f"{stem}.error.jsonl"
        elif user_save_path:
            args.save_path = user_save_path
            stem = os.path.splitext(user_save_path)[0]
            args.save_jsonl = user_save_jsonl or f"{stem}.jsonl"
            args.save_stem = stem
            args.error_path = f"{stem}.error.jsonl"
        else:  # 只给了 jsonl
            args.save_jsonl = user_save_jsonl
            stem = os.path.splitext(user_save_jsonl)[0]
            args.save_path = user_save_path or f"{stem}.json"
            args.save_stem = stem
            args.error_path = f"{stem}.error.jsonl"

    # 确保目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_jsonl), exist_ok=True)

    # 确保目录存在
    # os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    # os.makedirs(os.path.dirname(args.save_jsonl), exist_ok=True)
    # args.raw_data_dir = args.raw_data_dir + args.dataset_name + '/'
    # args.processed_data_dir = args.raw_data_dir.replace('raw', 'processed')
    # args.embeddings_dir = args.raw_data_dir.replace('raw', 'embeddings')
    # —— 数据目录规范化 ——
    if args.processed_data_dir:
        if not args.processed_data_dir.endswith('/'):
            args.processed_data_dir += '/'
        # raw/embeddings 仍按惯例使用 data/annotated/raw/<dataset> 与其同级的 embeddings
        args.raw_data_dir = os.path.join('data/annotated/raw/', args.dataset_name, '')
        base_embeddings_dir = args.raw_data_dir.replace('raw', 'embeddings')
    else:
        # 未显式传入时按旧逻辑推导
        args.raw_data_dir = os.path.join(args.raw_data_dir, args.dataset_name, '')
        args.processed_data_dir = args.raw_data_dir.replace('raw', 'processed')
        base_embeddings_dir = args.raw_data_dir.replace('raw', 'embeddings')

    # ★ 关键：根据 --knn_en/--en_mode 选择 embeddings 变体子目录，避免覆盖
    variant_tag = f"entity-{args.en_mode}" if args.knn_en else "sentence"
    args.embeddings_variant = variant_tag
    args.embeddings_dir = os.path.join(base_embeddings_dir, variant_tag)

    # 现在再创建目录（用的是带变体后缀的路径）
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.embeddings_dir, exist_ok=True)

    if not os.path.exists(args.processed_data_dir):
        os.makedirs(args.processed_data_dir)
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)

   # if args.model not in ["gpt-3.5-turbo", "gpt-4"] or "Llama-2-7b" not in args.model:#旧代码原本注释
    #    raise Exception("model not supported")#旧代码原本注释

    if args.prompt_type in ("ner", "ner_old", "ner_cot", "ner_cot2"):
        # 优先使用 4D 标签列；后续在保存/导出时若没拿到可回退到 2D
        args.gold_column = 'sentence_with_tags_4d'
        args.gold_column_fallback = 'sentence_with_tags'
    elif args.prompt_type in ('extract', 'extract_cot'):
        args.gold_column = 'list_extracted_skills'
    else:
        args.gold_column = 'sentence_with_tags'

    print("[ARGS.backend]", args.backend, "| model_name:", args.model_name)
    print("[SAVE]", args.save_path)
    return args



import importlib

def debug_load_and_print_prompts(args):
    """
    根据 args.prompt_module（或其它逻辑）动态加载
    PROMPT_TEMPLATES 和 RAG_PROMPTS，并打印 key 列表以供调试。
    """
    # 假设你在 parse_args 里把要用的模块名放到了 args.prompt_module
    prompt_module = getattr(args, "prompt_module", None)
    if prompt_module is None:
        # 根据 prompt_style 或默认决定模块名
        # 例如：
        if args.prompt_style:
            prompt_module = f"prompt_template_{args.prompt_style}"
        else:
            prompt_module = "prompt_template"  # 默认老的 prompt_template.py

    try:
        pm = importlib.import_module(prompt_module)
    except ImportError as e:
        print(f"[DEBUG] 无法导入 prompt 模块 `{prompt_module}`：{e}")
        return

    # 覆盖全局模板
    global PROMPT_TEMPLATES, RAG_PROMPTS
    if hasattr(pm, "PROMPT_TEMPLATES"):
        PROMPT_TEMPLATES = pm.PROMPT_TEMPLATES
    if hasattr(pm, "RAG_PROMPTS"):
        RAG_PROMPTS = pm.RAG_PROMPTS

    # 打印调试信息
    print(f"[DEBUG] Loaded prompt module: `{prompt_module}`")
    print(f"[DEBUG] PROMPT_TEMPLATES keys: {list(PROMPT_TEMPLATES.keys())}")
    if 'RAG_PROMPTS' in globals():
        print(f"[DEBUG] RAG_PROMPTS keys:    {list(RAG_PROMPTS.keys())}")
    else:
        print(f"[DEBUG] 没有检测到 RAG_PROMPTS")

    # （可选）打印当前要用的模板片段，检验一下结构
    ds = args.dataset_name if not args.no_specific else "all"
    tpl = PROMPT_TEMPLATES.get(ds, PROMPT_TEMPLATES.get("all"))
    print(f"[DEBUG] 使用的 PROMPT_TEMPLATES[\"{ds}\"]:\n{json.dumps(tpl, ensure_ascii=False, indent=2)}")

    if hasattr(globals(), "RAG_PROMPTS"):
        rp = RAG_PROMPTS.get(ds, RAG_PROMPTS.get("all"))
        print(f"[DEBUG] 使用的 RAG_PROMPTS[\"{ds}\"]:\n{json.dumps(rp, ensure_ascii=False, indent=2)}")


def _short_model_tag(args):
    """把模型名压缩成更短更可读的别名。"""
    if args.local_model:
        base = os.path.basename(args.local_model.rstrip('/'))
        # 常见模型做个映射；其余保留主干
        aliases = {
            "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
            "Qwen2.5-7B-Instruct":  "Qwen2.5-7B",
        }
        for k, v in aliases.items():
            if k in base:
                return v
        # LoRA/SFT 目录时给个标识
        if "lora" in base or "saves" in args.local_model:
            return f"SFT-{base}"
        return base
    # 云端/字符串模型名
    return (args.model or "base").replace('/', '_')

def _prompt_module_tag(args):
    """
    优先读取模块变量 PROMPT_MODULE_TAG: 'pm-old'/'pm-new'...
    回退策略：
      - 模块名以 _old 结尾 → 'old'
      - 其它包含 rag → 'new'
      - 否则 'old'
    """
    pm_name = getattr(args, "prompt_module", "prompt_template")
    try:
        import importlib
        mod = importlib.import_module(pm_name)
        tag = getattr(mod, "PROMPT_MODULE_TAG", None)  # e.g., 'pm-old'/'pm-new'
        if tag:
            # 允许直接写 'pm-old' 或 'old'
            tag = tag.replace("pm-", "")
            return tag
    except Exception:
        pass

    # 回退：看名字
    if pm_name.endswith("_old"):
        return "old"
    if "rag" in pm_name.lower():
        return "new"
    return "old"


def _prompt_style_tag(args):
    """
    返回 ps- 的短标签：
      generic/gen → 'gen'
      specific_old/spec_old → 'spec_old'
      其它/默认   → 'spec'
    """
    style = (getattr(args, "prompt_style", None) or "").lower()
    if style in {"generic", "gen"} or getattr(args, "no_specific", False):
        return "gen"
    if style in {"specific_old", "spec_old"}:
        return "spec_old"
    return "spec"


def _rag_tag(args):
    if not getattr(args, "rag", False):
        return None
    # 精简 RAG 标记：只保留后端和 topk（阈值通常可忽略）
    return f"rag-{args.rag_backend}-k{args.rag_topk}"

def _build_exp_tag(args):
    """精简后的文件名主干，例如：
       B1__md_Qwen2.5-14B__pm-old__ps-gen__s0
    """
    parts = []
    if args.exp_id:
        parts.append(args.exp_id)
    # 把 mdl- 前缀改成 md_，并且用上面缩短的模型别名
    parts.append(f"md_{_short_model_tag(args)}")
    parts.append(f"pm-{_prompt_module_tag(args)}")
    parts.append(f"ps-{_prompt_style_tag(args)}")
    parts.append(f"s{getattr(args, 'shots', 0)}")
    if getattr(args, "knn", False):
        parts.append("knn")
    if getattr(args, "knn_en", False):
        parts.append(f"en-{getattr(args, 'en_mode', 'anchor')}")
    if getattr(args, "classify", False):
        parts.append("cls")
    if "cot" in getattr(args, "prompt_type", ""):
        parts.append("cot")
    rtag = _rag_tag(args)
    if rtag:
        parts.append(rtag)
    if getattr(args, "cluster", None):
        parts.append(f"cluster-{args.cluster}")
    return "__".join(parts)

# 新增：把样本 sentence 做“实体锚定”的文本（供 knn_en 用）
def _anchor_sentence_from_sample(sample: dict) -> str:
    s = sample.get('sentence_with_tags') or sample.get('sentence') or ""
    # 1) 若已有 @@...## 标注，直接替换成 [SKILL]...[/SKILL]
    if '@@' in s and '##' in s:
        return s.replace('@@', '[SKILL] ').replace('##', ' [/SKILL]')
    # 2) 否则尝试用 BIO 恢复
    tokens = sample.get('tokens')
    tags = sample.get('tags_skill_clean') or sample.get('tags_skill') or sample.get('labels')
    if not (tokens and tags) or len(tokens) != len(tags):
        return sample.get('sentence', ' '.join(tokens or []))
    # BIO → 用方括号包住每个实体
    out = []
    i, n = 0, len(tokens)
    while i < n:
        if tags[i] == 'B':
            j = i
            while j+1 < n and tags[j+1] == 'I':
                j += 1
            span = " ".join(tokens[i:j+1])
            out.append(f"[SKILL] {span} [/SKILL]")
            i = j + 1
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

def main():
    args = parse_args()
    # parse_args() 里加：

    # —— Preprocess 阶段：按开关跳过 / 复用 / 重建 ——
    if args.skip_preprocess and not getattr(args, "force_preprocess", False):
        print(">>> skip_preprocess=True，跳过 preprocess_dataset。")
    else:
        for split in ['train', 'test']:
            processed_file = os.path.join(args.processed_data_dir, f'{split}.json')

            # 若不强制重建且已存在 processed 文件，则复用并跳过
            if os.path.exists(processed_file) and not getattr(args, "force_preprocess", False):
                print(f"[preprocess] 复用已存在文件：{processed_file}（如需重建，请加 --force_preprocess）")
                continue

            print(f"Processing {args.dataset_name} dataset, split={split}...")
            _ = preprocess_dataset(args, split)

    import importlib
    # 动态加载用户指定的 prompt_module，并替换 run 模块中的全局模板
    # pm = importlib.import_module(args.prompt_module)
    importlib.import_module(args.prompt_module)
    pm = args.prompt_module
    import run
    # ---------- 注入 Prompt 模块 ----------
    pm_mod = importlib.import_module(args.prompt_module)
    if hasattr(pm_mod, 'PROMPT_TEMPLATES'):
        run.PROMPT_TEMPLATES = pm_mod.PROMPT_TEMPLATES
        print(f"[DEBUG] run.PROMPT_TEMPLATES ← {args.prompt_module}.PROMPT_TEMPLATES")
    if hasattr(pm_mod, 'RAG_PROMPTS'):
        run.RAG_PROMPTS = pm_mod.RAG_PROMPTS
        print(f"[DEBUG] run.RAG_PROMPTS ← {args.prompt_module}.RAG_PROMPTS")

    # 打印模块标签（若有）
    pm_tag_dbg = getattr(pm_mod, "PROMPT_MODULE_TAG", None)
    print(f"[DEBUG] prompt module={args.prompt_module} tag={pm_tag_dbg or 'N/A'}")

    try:
        ds_key = args.dataset_name if not args.no_specific else "all"
        rp = run.RAG_PROMPTS.get(ds_key, run.RAG_PROMPTS.get("all", {}))
        instr = (rp.get("instruction", {}) or {}).get(args.prompt_type, "")
        print("[DEBUG] instruction preview:", (instr[:240] + "...") if isinstance(instr, str) else type(instr))
    except Exception as e:
        print("[WARN] prompt preview failed:", e)

    # ---------- 注入 Checking 模块 ----------
    import run  # 已有
    try:
        import prompt_check_template as chk_mod
        if hasattr(chk_mod, "CHECK_PROMPTS"):
            run.CHECK_PROMPTS = chk_mod.CHECK_PROMPTS
            print("[DEBUG] run.CHECK_PROMPTS ← prompt_check_template.CHECK_PROMPTS")
    except ImportError as e:
        print(f"[WARN] 未找到 prompt_check_template.py，检查环节将被跳过（--check 无效）。{e}")

    # debug_load_and_print_prompts(args)
    # —————— 2. classification 先行 ——————
    # args.data_path = os.path.join(args.processed_data_dir, 'test.json')
    # args.data_path = os.path.join(args.processed_data_dir, f'{args.split}.json')
    args.data_path = _resolve_split_json(args.processed_data_dir, args.split)
    print(f"[DATA] data_path → {args.data_path}")

    # —— 可选：为句子级样本注入段落上下文（混合策略）——
    paragraph_ctx = {}
    if args.paragraph_context_path and os.path.exists(args.paragraph_context_path):
        print(f"[context] Loading paragraph context from: {args.paragraph_context_path}")
        with open(args.paragraph_context_path, "r", encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                gid = (rec.get("global_id") or rec.get("id") or "").strip()
                para = (rec.get("text") or rec.get("paragraph") or rec.get("raw_text") or "").strip()
                if gid and para:
                    paragraph_ctx[gid] = para
        print(f"[context] paragraphs loaded: {len(paragraph_ctx)}")
    else:
        if args.paragraph_context_path:
            print(f"[context] WARN: file not found → {args.paragraph_context_path}")

    # 将 context 合并进当前 split 的样本
    if os.path.exists(args.data_path) and len(paragraph_ctx) > 0:
        df_for_ctx = pd.read_json(args.data_path)
        if "global_id" in df_for_ctx.columns:
            df_for_ctx["context"] = df_for_ctx["global_id"].map(lambda g: paragraph_ctx.get(str(g), ""))
            # 写一个带 context 的临时文件，并改写 data_path 指向它
            ctx_data_path = args.data_path.replace(".json", ".with_ctx.json")
            df_for_ctx.to_json(ctx_data_path, orient="records", force_ascii=False, indent=2)
            args.data_path = ctx_data_path
            print(f"[context] sentence samples enriched with paragraph context → {ctx_data_path}")

    args.demo_path = os.path.join(args.processed_data_dir, 'train.json')
    # ===== 段落模式：用 paragraph.jsonl 覆盖 data_path =====
    if getattr(args, "unit", "sent") == "paragraph":
        if not args.paragraph_path or not os.path.exists(args.paragraph_path):
            raise FileNotFoundError(
                f"[paragraph] 指定了 --unit paragraph，但未提供有效的 --paragraph_path：{args.paragraph_path}"
            )
        print(f"[paragraph] Loading paragraphs from: {args.paragraph_path}")
        para_rows = _load_paragraph_jsonl(args.paragraph_path)
        if not para_rows:
            raise RuntimeError("[paragraph] paragraph.jsonl 中没有有效段落记录。")

        # 写入一个临时 JSON 文件，让后续流程（分类/推理/评估）保持原样
        tmp_para_json = os.path.join(args.out_dir, "__paragraph_runtime.json")
        with open(tmp_para_json, "w", encoding="utf-8") as fw:
            json.dump(para_rows, fw, ensure_ascii=False, indent=2)
        args.data_path = tmp_para_json

        print(f"[paragraph] Prepared runtime dataset → {tmp_para_json}  (records={len(para_rows)})")
        # demo_path 仍然用句子级 train.json（不改），既可当 few-shot，也可当 RAG 的“示例库”

    # Download 阶段：若未显式 --process 且没有本地 processed 才尝试下载
    if not args.process:
        # 如果当前 split 对应的文件已经存在，就直接视为 processed_ok（分片场景）
        this_split_file = _resolve_split_json(args.processed_data_dir, args.split)
        processed_ok = os.path.exists(this_split_file)

        # 传统完整集（train/test 都在）也算 ok
        if not processed_ok and args.processed_data_dir:
            processed_ok = all(
                os.path.exists(os.path.join(args.processed_data_dir, f"{sp}.json"))
                for sp in ["train", "test"]
            )
        if not processed_ok:
            os.makedirs(args.raw_data_dir, exist_ok=True)
            for split in ['train', 'test']:
                raw_file = os.path.join(args.raw_data_dir, f'{split}.json')
                if not os.path.exists(raw_file):
                    print(f'Downloading {args.dataset_name} dataset, {split} split...')
                    download(args, split)
        else:
            print(f"[download] Found processed files under {args.processed_data_dir}, skip download.")

    # # Process 阶段 (拷贝或预处理)
    # for split in ['train', 'test']:
    #     print(f'Processing {args.dataset_name} dataset, split={split}...')
    #     _ = preprocess_dataset(args, split)

    if args.run and args.classify and args.backend != "local":
    # if args.classify:
    # if args.run and args.classify and args.backend != "local":
        # 拼分类结果文件名
        cls_path = args.save_path.replace(".json", "_cls.json")
        # 如果分类结果已存在，且不需要强制重跑，就直接返回
        if os.path.exists(cls_path) and not args.force_reclassify:
            print(f"▶ Found existing classification file {cls_path}, skip classification pass.")
        else:
            print("▶ Running classification pass …")
            df_data = pd.read_json(args.data_path)
            if args.sample and args.sample > 0:
                #df_data = df_data.sample(n=min(args.sample, len(df_data)), random_state=1450).reset_index(drop=True)#随机样本
                df_data = df_data.head(args.sample).reset_index(drop=True)#固定样本
            classifications = {}
            for _, row in df_data.iterrows():
                sid = row["id"]
                sent = row["sentence"]
                msgs = [
                    {"role": "system", "content": CLASSIFICATION_PROMPT["system"]},
                    {"role": "user", "content": CLASSIFICATION_PROMPT["user"].format(sentence=sent)}
                ]
                print(f"\n[Classify Prompt ID={sid}]\n" + json.dumps(msgs, ensure_ascii=False, indent=2))
                print("saf_chat参数","msgs:",msgs,"args:",args)
                resp = safe_chat(msgs, args)
                label = resp["choices"][0]["message"]["content"].strip()
                label = _norm_cls_label(label)  # ← 新增
                classifications[sid] = label
                print(f"  [Classify] ID={sid} → {label}")
                classifications[sid] = label

            # 写盘
            with open(cls_path, "w", encoding="utf-8") as f:
                json.dump(classifications, f, ensure_ascii=False, indent=2)
            print(f"→ Saved {len(classifications)} labels to {cls_path}")

        # 如果只是分类模式就退出
        if not args.run and not args.eval:
            return

    #
    #——— 为 ESCO 生成 KNN embeddings ———
    # ——— 为 ESCO 生成 KNN embeddings ———
    if args.knn:
        for split in ['train', 'test']:
            emb_save_path = os.path.join(args.embeddings_dir, f'{split}.pt')

            # 已存在且未要求强制重建 → 直接复用
            if os.path.exists(emb_save_path) and not args.force_reembed:
                print(f'[embed] Reuse existing embeddings: {emb_save_path}')
                continue

            print(f'[embed] Generating dataset {split} embeddings → {emb_save_path}')
            # 从 processed JSON 读出 samples
            source_path = os.path.join(args.processed_data_dir, f'{split}.json')
            source_dataset = read_json(source_path)

            # ★ 根据 --knn_en 选择文本：实体锚定 or 句级
            if args.knn_en:
                dataset_texts = [_anchor_sentence_from_sample(sample) for sample in source_dataset]
            else:
                dataset_texts = [
                    sample.get('sentence', ' '.join(sample.get('tokens', [])))
                    for sample in source_dataset
                ]

            dataset_ids = [sample.get('id', sample.get('idx')) for sample in source_dataset]

            # 生成并落盘
            # 只有需要 kNN few-shot 才进行 demo embedding
            if getattr(args, "shots", 0) > 0 and getattr(args, "knn", False):
                dataset_embed = embed_demo_dataset(dataset_texts, args.dataset_name)
            else:
                dataset_embed = None
            # dataset_embed = embed_demo_dataset(dataset_texts, args.dataset_name)
            torch.save({'embeddings': dataset_embed, 'ids': dataset_ids}, emb_save_path)
            print(f'[embed] Saved embeddings to {emb_save_path}')

    if args.run:
        # Load dataset
        dataset = pd.read_json(args.data_path)
        if args.run and getattr(args, "ctx_assist", False):
            # 兼容列名：sentence_order/sent_id；title/招聘岗位；global_id/Global_ID
            if "sentence_order" not in dataset.columns and "sent_id" in dataset.columns:
                dataset["sentence_order"] = dataset["sent_id"]
            if "title" not in dataset.columns and "招聘岗位" in dataset.columns:
                dataset["title"] = dataset["招聘岗位"]
            if "global_id" not in dataset.columns and "Global_ID" in dataset.columns:
                dataset["global_id"] = dataset["Global_ID"]

            def _mk_ctx(df_g, w):
                df_g = df_g.sort_values("sentence_order").reset_index(drop=True)
                sents = df_g["sentence"].tolist()
                prevs, nexts = [], []
                for i in range(len(sents)):
                    p = " ".join(sents[max(0, i - w):i])
                    n = " ".join(sents[i + 1:i + 1 + w])
                    prevs.append(p)
                    nexts.append(n)
                df_g["ctx_prev"] = prevs
                df_g["ctx_next"] = nexts
                return df_g

            if "global_id" in dataset.columns and "sentence_order" in dataset.columns:
                dataset = (dataset
                           .groupby("global_id", group_keys=False)
                           .apply(lambda g: _mk_ctx(g, w=max(1, int(args.ctx_window)))))
            else:
                # 字段缺失时，至少提供空上下文
                dataset["ctx_prev"] = ""
                dataset["ctx_next"] = ""

        if args.exclude_empty:
            dataset['has_item'] = dataset.apply(lambda row: len(row['skill_spans'])>0, axis=1)
            dataset = dataset[dataset['has_item'] == True]
            dataset.drop(columns=['has_item'], inplace=True)
        print(f"Loaded {len(dataset)} examples from {args.dataset_name} dataset.")
        if len(dataset['id']) != len(set(dataset['id'].values.tolist())):
            raise Exception("The ids are not unique")
        # Run inference
        if args.sample != 0:
            sample_size = min(args.sample, len(dataset))
            dataset = dataset.sample(sample_size, random_state=1450).reset_index(drop=True)
        run_openai(dataset, args)

    # main.py 末尾
    if args.eval:
        from eval_adapter import make_eval_file
        print(f"Evaluating {args.save_path}...")
        # processed_test_path = os.path.join(args.processed_data_dir, 'test.json')
        # processed_test_path = os.path.join(args.processed_data_dir, f'{args.split}.json')
        processed_test_path = _resolve_split_json(args.processed_data_dir, args.split)
        task_type = 'extract' if args.prompt_type.startswith('extract') else 'ner'
        packed_eval_path = f"{args.save_stem}.eval_{task_type}.json"  # ← 这里必须是纯字符串
        pack_stats = make_eval_file(
            pred_path=args.save_path,
            processed_test_path=processed_test_path,
            out_path=packed_eval_path  # ← 传入字符串
        )
        all_metrics = eval(packed_eval_path)
        print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()