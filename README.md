# LLM 技能抽取框架 (LLM Skill Extraction Framework)

## 概述

本项目是一个用于从招聘广告 (Job Advertisements) 中使用大型语言模型 (LLM) 提取技能的综合性、模块化框架。该框架专为基准测试和推理而设计，支持多种公开数据集、高级提示策略（如 RAG 和 CoT）以及灵活的推理后端。

系统围绕一个中心运行器 (`main.py`) 构建，该运行器负责协调数据预处理、提示词生成、LLM 推理（包括云端 API 和本地模型）、(可选的) 预分类、kNN/RAG 检索以及最终的评估。

## 核心特性

* **多数据集支持**: 包含为多个国际技能抽取基准定制的提示词模板，例如：
    * `chinese_skillspan` (中文, LKST 四维)
    * `skillspan` (英文, Hard/Soft skills)
    * `fijo` (法文)
    * `gnehm` (德文, 仅 ICT 技能)
    * `kompetencer` (丹麦文)
    * `sayfullina` (英文, 仅 Soft skills)
* **高级提示策略**: 支持多种任务格式，包括：
    * **NER 标注**: 在原句中插入 `@@...##` 标签 (例如 `ner`, `ner_old`)。
    * **列表抽取**: 将技能作为列表或 JSON 提取 (例如 `extract`)。
    * **思维链 (CoT)**: 指示模型先思考再回答 (例如 `ner_cot`)。
* **RAG 与 kNN 增强**:
    * `--knn`: 启用 kNN，从训练集中检索相似的 Few-shot 示例注入提示词。
    * `--rag`: 启用 RAG，从外部知识库检索相关的技能定义或背景知识。
* **灵活的推理单元**: 支持按句子 (`--unit sent`) 或按完整段落 (`--unit paragraph`) 进行推理。
* **预分类 (Pre-classification)**: 可选的 (`--classify`) 预处理步骤，在提取前先判断句子是否包含技能，以优化流程或过滤无关文本。
* **SFT 数据转换**: 提供 `convert_to_lkst_alpaca.py` 实用工具，可将 Doccano 导出的标注数据转换为 Alpaca 格式，用于监督微调 (SFT)。

## 安装与设置

1.  **克隆仓库**:
    ```bash
    git clone ...
    cd llm-skill-extraction
    ```

2.  **安装依赖**:
    * 项目依赖于 `pandas`, `openai`, `torch` 等库。
    * (推荐) 创建虚拟环境并安装:
        ```bash
        pip install -r requirements.txt
        ```

3.  **API 密钥**:
    * 对于 OpenAI 等云端模型，请在 `api_key.py` 文件中设置您的 `OPENAI_API_KEY`，或将其设置为环境变量。

## 使用指南

### 步骤 1: 数据准备

**A) 推理与评估**

1.  将您的原始数据集 (如 `train.json`, `test.json`) 放置在 `data/annotated/raw/<dataset_name>/` 目录下。
2.  首次运行时，`main.py` 会自动处理数据，并将处理后的文件保存在 `data/annotated/processed/<dataset_name>/`。
3.  您也可以使用 `--force_preprocess` 强制重新处理数据。

**B) 模型微调 (SFT)**

如果您需要为 `chinese_skillspan` (LKST) 任务微调模型，请使用 `convert_to_lkst_alpaca.py` 转换您的 Doccano 标注数据。

```bash
python convert_to_lkst_alpaca.py \
    --input /path/to/your_doccano_export.jsonl \
    --output /path/to/alpaca_train_data.jsonl
```

该脚本会将 Doccano 的 `label` 或 `entities` 字段转换为 Alpaca 格式，包含固定的指令 (instruction)、输入 (input) 和期望的带标签输出 (output)。

### 步骤 2: (可选) 生成 kNN 嵌入

如果使用 `--knn` 功能，框架会自动为 `train.json` 生成并缓存句子嵌入。

  * 嵌入文件将保存在 `data/annotated/embeddings/<dataset_name>/...` 路径下。
  * 使用 `--force_reembed` 可强制重新生成嵌入。

### 步骤 3: 运行推理与评估

使用 `main.py` 脚本发起运行。

**示例命令 (chinese\_skillspan, gpt-4o, CoT 提示, kNN, RAG)**:

```bash
python main.py \
    --dataset_name chinese_skillspan \
    --model gpt-4o \
    --prompt_module prompt_template_rag \
    --prompt_type ner_cot \
    --split test \
    --shots 5 \
    --knn \
    --rag \
    --rag_topk 3 \
    --run \
    --eval
```

**示例命令 (段落级推理)**:

```bash
python main.py \
    --dataset_name chinese_skillspan \
    --model gpt-4o \
    --prompt_type ner \
    --unit paragraph \
    --paragraph_path /path/to/my_paragraphs.jsonl \
    --run \
    --eval
```

### 关键参数 (`main.py`)

  * `--dataset_name`: 要使用的数据集名称 (例如 `chinese_skillspan`, `skillspan`)。
  * `--model`: 要使用的云端模型 (例如 `gpt-4o`, `gpt-3.5-turbo`)。
  * `--local_model`: (当 `--backend local` 时) 本地模型的路径或 repo id。
  * `--prompt_module`: 包含提示词模板的 Python 模块 (例如 `prompt_template_rag`)。
  * `--prompt_type`: 要使用的具体提示词键名 (例如 `ner`, `ner_cot`, `extract`)。
  * `--run`: 执行推理运行。
  * `--eval`: 在运行后执行评估。
  * `--split`: 要处理的数据分片 (例如 `test`, `dev`, 或 `dev.p000`)。
  * `--save_path`: 指定保存结果的 `.json` 文件路径。
  * `--save_stem`: 指定保存路径的词干 (会自动派生 `.json`, `.jsonl` 等)。

**功能开关**:

  * `--classify`: 启用推理前的技能/非技能二分类步骤。
  * `--rag`: 启用 RAG (从知识库检索)。
  * `--knn`: 启用 kNN (从训练集检索 Few-shot 示例)。
  * `--shots <N>`: 指定 Few-shot 示例的数量 (0 表示 Zero-shot)。
  * `--unit <sent|paragraph>`: 推理单元是句子还是段落。
  * `--paragraph_path <path>`: 当 `unit=paragraph` 时，指定输入的 `.jsonl` 文件。

## 核心组件

  * **`main.py`**:
      * **描述**: 项目的主入口和协调器。
      * **功能**: 解析参数；加载数据；(可选) 预分类；(可选) 生成 kNN 嵌入；调用 `run_openai` 执行推理；调用评估脚本。
  * **`prompt_template_rag.py`**:
      * **描述**: 核心的提示词库。
      * **功能**: 以 Python 字典形式存储，为每个数据集 (`chinese_skillspan`, `fijo` 等) 和每种任务类型 (`ner`, `extract`, `ner_cot`) 定义了精确的 `system` 消息、`task_definition` 和 `instruction`。
  * **`convert_to_lkst_alpaca.py`**:
      * **描述**: SFT 数据转换工具。
      * **功能**: 读取 Doccano 导出的 JSONL 文件，将其 `label` (三元组) 或 `entities` (对象列表) 字段中的标注，转换为符合 `chinese_skillspan` LKST 规范的 Alpaca 格式 (instruction, input, output)，其中 output 是带 `@@片段##[L|K|S|T]` 标签的句子。

<!-- end list -->

```
```
