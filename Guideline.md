| Rule 规则 | Description 描述 | Example 例子 |
|---|---|---|
| 1 | Scope filtering；范围过滤 | Annotate only responsibilities/requirements blocks; skip slogans/titles/benefits/age/shift/salary.<br>仅在职责/工作内容/任职要求区块标注；口号、岗位名、待遇、年龄、班次、薪酬等一律不标。 |
| 2 | Skill signals only；仅技能要素 | From "Responsible for routine work, coordinate milestones, produce reports", label only concrete actions.<br>例：原文“负责日常工作，推进里程碑的沟通协调、报表制作。”→ 推进里程碑的沟通协调 → T，报表制作 → S。 |
| 3 | Label set (SKTL)；标签集（SKTL） | K=Knowledge；S=Skills；T=Transversal；L=Language。<br>Examples: CET-6 (L); Statistics (K); Process design (S); Communication skills (T)。<br>例：英语六级 → L；统计学 → K；流程制定 → S；沟通能力 → T。 |
| 4 | Flat, non-nested, non-overlapping；平面、不嵌套、不重叠 | Multiple spans per sentence allowed, each a contiguous substring; no overlap/nesting; no verb-copying to fabricate spans.<br>同句可多跨，但须原文连续、互不相交；禁止复制动词人造片段。 |
| 5 | Minimal yet sufficient (context-first)；最小且语义充分（语境优先） | Choose the shortest contiguous piece that still self-identifies as L/S/K/T; avoid hard splitting that causes drift.<br>取能独立判断 L/S/K/T 的最短连续片段；避免硬拆致标签漂移。 |
| 6 | Remove function/hedge words；去冗余虚词 | Drop polite/degree/modality fillers; keep core action/object. "Be able to do data visualization" → "data visualization" (S)。<br>删“请/较强的/能够/开展/相关/其”等；例：“能够熟练地进行数据可视化。”→ 数据可视化 → S；“设计数据模型。”→ 数据模型设计 → S。 |
| 7 | Parallel items: may split, do not over-split；并列可拆，但不硬拆 | With ", / and / or", split only if each item is self-contained. "design data models; build metric system; deliver reports" → three S。<br>例：“设计数据模型、搭建指标体系、出具分析报告。”→ 数据模型设计 → S；指标体系搭建 → S；分析报告出具 → S。 |
| 8 | Scene modifiers outside span；场景修饰不入段 | Ignore "internal/external/online/offline/cross-department", unless it forms the entity itself。<br>例：“线上开展数据可视化培训。”→ 数据可视化培训 → S。 |
| 9 | Versions/IDs in span only if ontological；版本/编号仅为本体时保留 | Keep standard IDs (ISO 27001; RFC 2616) as K; drop software version numbers if not essential。<br>例：ISO 27001 → K；RFC 2616 → K；“Python 3.11、MySQL 8”→ Python → K / MySQL → K（版本号不入段）。 |
| 10 | Context-first + fallback **L > S > K > T**；语境优先 + 兜底优先级 | Decide by context (language? action? knowledge? soft-skill?). If unsure, apply L > S > K > T。<br>先判语境；仍不明确按 L > S > K > T 落位。 |
| 11 | Encoding/offsets (Doccano)；编码与偏移 | Start index = 0; end is exclusive; export converters must keep offsets; spans are verbatim substrings。<br>起点 0、end 右开；导出转换偏移一致；跨度严格取自原文、禁止拼接。 |
| 12 | K — Knowledge (ontology/standards/qualifications)；K — 知识（本体/标准/资格） | Quick test: entity/standard/framework/principle/model names as entities; held qualifications。<br>例：ISO 27001 → K；英国/欧盟驾驶执照 → K。 |
| 13 | S — Skills (executable duties/actions)；S — 技能（可执行职责/动作） | Verb + object/output; using tools/languages/platforms; executing to standards。<br>例：使用 Excel 进行数据分析 → S；用 Python 训练和部署模型 → S；使用 HALCON/VisionPro 进行视觉检测 → S。 |
| 14 | T — Transversal (soft skills)；T — 通用/软能力 | Cross-role traits (communication, learning, stress tolerance, responsibility). If phrased as actions, label S。<br>例：沟通协调能力 → T；若写成“沟通协调客户诉求并推进解决”→ 沟通协调客户诉求并推进解决 → S。 |
| 15 | L — Language (ability/level/tests)；L — 语言（能力/等级/考试） | Languages, levels, exams, speaking/writing/reading/listening; communication acts can be separate S。<br>例：英语 CET-6 → L。 |
| 16 | Conflict: K vs S；冲突：K 与 S | Execution/output/roll-out → S; entity/standard-as-entity → K; if unsure, prefer S。<br>例：“熟悉通信协议”→ 通信协议 → K；“制定视觉流程”→ 视觉流程制定 → S。 |
| 17 | Conflict: L vs S；冲突：L 与 S | Language ability/level/certificate → L; using language to communicate/write → S (may split)。<br>例：“用英语与海外客户邮件沟通”→ 英语 → L；邮件沟通 → S。 |
| 18 | Conflict: L vs K；冲突：L 与 K | Natural language → L; theoretical/terminological items → K; programming-language certificates → K。<br>例：“英语六级”→ CET-6 → L；“OCJP-Java”→ OCJP-Java 认证 → K。 |
| 19 | Conflict: T vs S；冲突：T 与 S | Executable behaviors (coordinate/write/organize) → S; general traits → T。<br>例：“主动沟通协调”→ 沟通协调 → S；“具备沟通能力”→ 沟通能力 → T。 |
| 20 | Conflict: S contains K object；冲突：S 中包含 K 对象 | Keep S only (no double-tag/no extra K). Executing to a standard stays S。<br>例：“按 ISO 9001 开展审核”→ 按 ISO 9001 开展审核 → S（不另起 K）。 |
| 21 | Conflict: T with actions or K objects；冲突：T 与动作/K 组合 | Plain T or parallel traits → T; “T + concrete action/output” → S。<br>例：“结果导向”→ 结果导向 → T；“结果导向地提升转化率”→ 提升转化率 → S。 |
| 22 | Key example A: coordinate cross-teams → S；关键示例 A：统筹跨团队 → S | "Coordinate product/marketing/operations/commercialization resources" → one S span (objects are part of the same duty)。<br>“统筹协调产品、市场、运营、商业化团队资源”→ 整段标为 S。 |
| 23 | Key example B: responsible for X, Y, Z → S；关键示例 B：负责 X/Y/Z → S | Scope nouns remain within the same S; do not split into S+K+K。<br>“负责流程制定、市场、法务”→ 整段标为 S（勿拆成 S+K+K）。 |
| 24 | Key example C: English + email + Excel → L/S/S；关键示例 C：英语+邮件+Excel → L/S/S | English (L); email communication (S); use Excel to perform data aggregation (S)。<br>“英语” → L；“邮件沟通” → S；“用 Excel 进行数据汇总” → S。 |
| 25 | Key example D: licence (K) + comply (S)；关键示例 D：执照(K)+遵守(S) | "Driving licence (UK/EU)" (K); "comply with health, safety and audit standards" (S)。<br>“驾驶执照(英/欧)” → K；“遵守健康与安全及审计标准” → S。 |
| 26 | Key example E: soft skills → T/T；actions → S；关键示例 E：软能力 T/T；动作归 S | "Communication skills" (T); "coordinate client requests and drive resolution" (S)。<br>“沟通协调能力” → T；“沟通客户诉求并推进解决” → S。 |
| 27 | Tools/languages used to do tasks → S；使用工具/语言完成任务 → S | "Deploy models with Python" (S); "write reports in English" → English (L) + "write reports" (S)。<br>“使用 Python 部署模型” → S；“以英文撰写报告”→ “英语” → L + “撰写报告” → S。 |
| 28 | Standard as entity = K；executing to it = S；标准本体=K；按之执行=S | "Familiar with ISO 9001 system" (K); "conduct audits per ISO 9001" (S)。<br>“了解 ISO 9001 体系” → K；“按 ISO 9001 审核” → S。 |
| 29 | Export fields & splits；导出字段与数据划分 | Export text and spans{start,end,label,text}; keep fixed IID/OOD splits for evaluation。<br>导出 text、spans{start,end,label,text}；保持固定 IID/OOD 划分用于评测。 |
| 30 | QC metrics & sampling；质检指标与抽检 | Strict/Relaxed Span-F1, Cohen's Kappa; every 200 items, double-annotate 10% and retrain if >10% disagreement。<br>采用 Strict/Relaxed Span-F1、Cohen's Kappa；每 200 条抽 10% 双标复核，分歧 >10% 回溯培训。 |

