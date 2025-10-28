import json
import argparse
from typing import List, Dict, Tuple

# 固定的指令（instruction）。保持和后面推理/评估一致，非常关键。
LKST_INSTRUCTION = """你是一名熟悉中文招聘文本的人力资源与技能本体（ESCO）专家。你的任务是在原句中标注与能力相关的片段，并严格遵循 ESCO-1.20 的 LKST 平面标注规范：

• [L] Language：仅限自然语言能力/证书/等级/使用（如“英语六级”“日语N2”“普通话二甲”“能用英语沟通”）。编程语言或技术栈的使用不算 L。
• [K] Knowledge：学科/领域/规范/标准/法规/框架/工具体系等“拥有/了解/熟悉/掌握……知识/原理/规范”的知识客体（名词本体）。
• [S] Skills：可训练、可执行、可操作的能力或方法（动作或过程类，如“制定/设计/评审/配置/开发/调试/部署/维护/优化/调优”等；编程语言和工具使用也归 S）。
• [T] Transversal：跨岗位通用能力（沟通、协作、学习、时间管理、抗压、领导力、客户导向等软技能）。

全局标注规则：
1. 平面标注：不重叠、不嵌套；并列项逐一切分。
2. 最小充分片段：去掉“参与/负责/进行/熟悉/掌握/具备/能够/良好/较强”等触发或评价词，只保留能自解释的核心片段。
3. K/S 判定：名词本体处于“知识/了解/熟悉/具备…知识/理解”语境多判 K；处于“制定/设计/评审/配置/开发/调试/部署/维护/优化/调优”等动作语境多判 S。
4. L 仅限自然语言能力（等级/证书/使用），编程语言或工具使用请判为 S。
5. 冲突兜底：若片段在多类之间仍难区分，按优先级 L > S > K > T 进行唯一分类。
6. 如果该句没有任何能力相关内容，请原样返回。

输出要求：
- 在原句中用 `@@片段##[L|K|S|T]` 形式包住每个能力片段。
- 输出只包含一行：标注后的整句。"""

# 标签映射：把 doccano 导出的标签规范化到 {L,K,S,T}
LABEL_MAP = {
    "L": "L",
    "K": "K",
    "S": "S",
    "T": "T",
    "SILVER_L": "L",
    "SILVER_K": "K",
    "SILVER_S": "S",
    "SILVER_T": "T",
    "LANGUAGE": "L",
    "KNOWLEDGE": "K",
    "SKILL": "S",
    "TRANSVERSAL": "T",
}

def normalize_entities(rec: Dict) -> List[Tuple[int,int,str]]:
    """
    统一把一条样本里的标注实体拉成 [(start, end, mapped_label), ...]
    支持两种输入格式：
    1) rec["label"] = [[start,end,label_str], ...]
    2) rec["entities"] = [{"start_offset":...,"end_offset":...,"label":...}, ...]
    返回时会去重同一 (start,end,label) 的条目
    """

    spans = []

    # 格式1：label = [[start,end,label_str], ...]
    if "label" in rec and isinstance(rec["label"], list):
        for triplet in rec["label"]:
            if len(triplet) != 3:
                continue
            start, end, raw_lab = triplet
            mapped = LABEL_MAP.get(raw_lab)
            if mapped is None:
                # 未知类型，保守地映射为 "S"（可改策略）
                mapped = "S"
            spans.append((start, end, mapped))

    # 格式2：entities = [{"start_offset":..., "end_offset":..., "label":...}, ...]
    if "entities" in rec and isinstance(rec["entities"], list):
        for ent in rec["entities"]:
            start = ent.get("start_offset")
            end   = ent.get("end_offset")
            raw_lab = ent.get("label")
            if start is None or end is None or raw_lab is None:
                continue
            mapped = LABEL_MAP.get(raw_lab)
            if mapped is None:
                mapped = "S"
            spans.append((start, end, mapped))

    # 去重相同 (start,end,label)
    spans = list(set(spans))

    # 按 start 升序（如果重叠，按start排序后插入就会打出嵌套/交叉，
    # 但你说你们标注是平面不重叠，所以这里默认不会冲突）
    spans.sort(key=lambda x: x[0])

    return spans

def insert_tags(text: str, spans_sorted: List[Tuple[int,int,str]]) -> str:
    """
    把 spans_sorted (start,end,label_type) 依次插回原句，生成带 @@...##X 的 output 字符串。
    要求：spans_sorted 已按 start 升序，且不重叠。
    """

    if not spans_sorted:
        return text  # 没有标注，直接原句

    out_chunks = []
    cursor = 0

    for (start, end, lab) in spans_sorted:
        # 先把游标到span开始前的原文写进去
        if cursor < start:
            out_chunks.append(text[cursor:start])
        # span正文
        span_text = text[start:end]
        out_chunks.append(f"@@{span_text}##{lab}")
        cursor = end

    # 最后剩余部分
    if cursor < len(text):
        out_chunks.append(text[cursor:])

    return "".join(out_chunks)

def convert_file(input_path: str, output_path: str):
    """
    把 Doccano 导出的 (id, text, meta, label / entities) 格式
    转成 LLaMA-Factory 可用的 Alpaca SFT 样本格式。
    """

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            rid  = rec.get("id")
            text = rec["text"]
            meta = rec.get("meta", {})

            spans = normalize_entities(rec)  # [(start,end,label), ...]
            tagged_text = insert_tags(text, spans)

            out_obj = {
                "instruction": LKST_INSTRUCTION,
                "input": text,
                "output": tagged_text,
                "id": rid,
                "meta": meta,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="doccano导出的jsonl（含label或entities）")
    ap.add_argument("--output", required=True, help="要写出的alpaca训练jsonl")
    args = ap.parse_args()

    convert_file(args.input, args.output)

if __name__ == "__main__":
    main()
