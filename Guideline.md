| 规则 Rule | 描述 Description                                                      | 例子 Example                                                                         |
| -- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 1  | 不嵌套；*No nesting*                                                | 错误：在“数据分析”中再标“分析”。 *Wrong: label “analysis” inside “data analysis”.*               |
| 2  | 不重叠；*No overlap*                                                | 错误：两个跨度共享字符。 *Wrong: two spans share characters.*                                  |
| 3  | 平面标注，多跨度但互不包含；*Flat, multiple disjoint spans*                   | 一句话提取多个连续、不重叠片段。 *Multiple disjoint contiguous spans from one sentence.*           |
| 4  | 跨度必须原文连续；禁止拼接；*Contiguous only; no fabrication*                 | 禁止复制动词去补名词碎片。 *Do not duplicate verbs to attach to noun scraps.*                   |
| 5  | 口号/愿景/礼貌语/岗位名/待遇等不标；*Slogans/titles/benefits unlabelled*        | “公司使命：改变世界”不标。 *“Our mission: change the world” → no label.*                       |
| 6  | 笼统职责不单独成跨度；*Generic catch-alls not spans*                       | 只标后续具体动作/对象。 *Label the specific action/object only.*                              |
| 7  | 无具体对象或动作则跳过；*Skip if no concrete action/object*                 | “负责相关工作”→跳过。 *“Responsible for related work” → skip.*                              |
| 8  | 最小但语义充分；可去壳词；*Minimal yet self-contained; drop hedges*          | “具备较强的沟通协调能力”→“沟通协调能力”。 *“Strong communication” → “communication”.*                |
| 9  | 禁剪残词；*No partial morphemes*                                     | 错：“统筹协”“项目落”。 *Wrong: “coordi-”, “-deliver”.*                                      |
| 10 | 仅可“安全拆”；*Split only when safe*                                  | “英语”(L) 与 “邮件沟通”(S) 可拆。 *Split “English”(L) and “email communication”(S).*         |
| 11 | 标签冲突先看语境；不行则 L>S>K>T；*Context first; fallback L>S>K>T*          | “按GMP执行质检”→S。 *“Inspect per GMP” → S.*                                             |
| 12 | **K**：知识/标准/资格本体；*K: bodies/standards/qualifications*           | “GMP”“驾驶执照(英/欧)”。 *“GMP”, “driving licence (UK/EU)”.*                              |
| 13 | **S**：执行/职责/用工具完成任务；*S: execution/tasks/tools*                  | “SQL抽取数据”。 *“Extract data with SQL”.*                                              |
| 14 | **T**：通用素质；写成动作则判S；*T: soft skills; actional → S*               | “沟通协调能力”(T)；“沟通客户诉求并推进”(S)。 *“Communication”(T); “coordinate client requests”(S).* |
| 15 | **L**：语言/等级/考试/证书；*L: language/level/tests/certs*               | “英语/CET-6/IELTS”。 *“English/CET-6/IELTS”.*                                         |
| 16 | 例：统筹协调资源→整段S；*Example: coordinating resources → S*              | “统筹协调产品、市场、运营、商业化团队资源”(S)。 *Keep whole span as S.*                                 |
| 17 | 例：负责流程制定、市场、法务→整段S；*Example: responsible for X,Y,Z → S*         | 不拆成S+K+K。 *Do not split into S+K+K.*                                               |
| 18 | 例：英语+邮件沟通+Excel汇总→L/S/S；*English + email comms + Excel → L/S/S* | “英语”(L)；“邮件沟通”(S)；“用Excel进行数据汇总”(S)。 *All split safely.*                           |
| 19 | 例：执照(K)+遵守标准(S) 可拆；*Licence(K) + comply(S) split*               | “驾驶执照(英/欧)”(K)；“遵守健康与安全及审计标准”(S)。 *Split by role.*                                 |
| 20 | 例：沟通协调能力/承压能力→T/T；*Communication/pressure tolerance → T/T*      | 若写成具体动作则S。 *Action phrasing → S.*                                                  |
| 21 | 工具/语言用于完成任务→S；*Using tools/languages to do → S*                 | “使用Python部署模型”(S)；“以英文撰写报告”→“英语”(L)+“撰写报告”(S)。                                     |
| 22 | 标准本体=K；按标准执行=S；*Standard entity=K; executing=S*                 | “了解ISO 9001”(K)；“按ISO 9001审核”(S)。 *Map by role.*                                   |
| 23 | 拆分不得改变原意或致歧义；*Splits must preserve meaning*                     | 拆后“市场”变模糊→不要拆。 *If “market” becomes ambiguous → keep whole.*                       |
| 24 | 标注顺序：L→S→K → T；*Order: L → S → K → T*                               | 先语言，再职责，再区分知识/软素质。 *Prioritize L, then S, then K/T.*                               |
| 25 | 自检：无重叠/无嵌套；*Check: no overlap/nesting*                          | 检查相邻跨度边界。 *Verify boundaries.*                                                     |
| 26 | 自检：禁把职责拆成名词堆；*Check: no noun-scrap splitting*                   | “统筹A、B、C”勿拆成A/B/C。 *Do not split into bare nouns.*                                 |
| 27 | 自检：防S被误切成K；*Check: avoid turning S into K*                      | 参见例1/2。 *See Ex. 16/17.*                                                           |
| 28 | 自检：每跨原文连续且未拼接；*Check: contiguous, unfabricated*                 | 需跨越不连续文本时放弃拆分。 *If discontinuous, don’t split.*                                    |
| 29 | 自检：合规表达判S；*Check: compliance phrasing → S*                      | “按GMP流程质检”→S。 *“Inspect per GMP” → S.*                                             |
| 30 | 自检：不确定用 L>S>K>T 兜底；*Check: fallback L>S>K>T*                    | “英文会议沟通”→L+S；“学习GMP体系”→K。 *Map consistently.*                                      |

