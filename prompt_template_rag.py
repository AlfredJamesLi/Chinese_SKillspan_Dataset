RAG_PROMPTS = {
    "all": {
        "system": (
            "You are an expert human resource manager. "
            "Your task is to analyze job descriptions and identify all the skills required from the candidate."
            "\nAlways respond in the same language as the input sentence; do not translate."
        ),
        "system_by_prompt": {
            "ner_old": "You are an expert human resource manager. You need to analyse skills required in job offers"
        },
        "task_definition": (
            "Task Definition: Extract every skill or competency that the candidate must possess, "
            "including technical skills, domain knowledge, and soft skills. "
            "Skills are defined as any ability, tool, or behavior needed to perform the job. "
            "Output format: Either wrap each item with @@ and ##, e.g. @@Python programming## "
        ),
        # 以下三个字段都是占位，运行时填入实际检索到的文本列表
        # general_bg: 通用背景知识列表
        # skill_defs: 技能定义列表
        # demos: few-shot 示例列表，元素格式为 (input, output)
        "background": (
            "[General Background]\n---\n{general_bg}\n---"
        ),
        "definitions": (
            "[Skill Definitions]\n---\n{skill_defs}\n---"
        ),
        "demonstrations": (
            "[Demonstrations]\n{demos}"
        ),
        "instruction": {
            "ner": (
                "You are given a sentence from a job description. "
                "Replicate the sentence and highlight all the skills and competencies "
                "(including technical skills, domain knowledge, and soft skills) "
                "by surrounding them with tags @@ and ##. "
                "If the sentence contains no such elements, copy the sentence unchanged."
            ),
            "ner_old": (
                "You are given a sentence from a job description. Replicate the sentence and highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically."
            ),
            "ner_cot": (
                "You are given a sentence from a job description. "
                "Identify all required skills/competencies (technical, domain, and soft). "
                "Briefly think step by step internally, then OUTPUT ONLY the final annotated sentence with @@ and ##. "
                "If none, copy the sentence unchanged. Do NOT reveal your reasoning."
            ),
            "extract": (
                "You are given a sentence from a job description. "
                "Extract every skill or competency required from the candidate as a list, one skill per line. "
                "If no skills are present in the sentence, return \"None\"."
            )
        }
    },
    "skillspan": {
        "system": (
            "You are an expert human-resource manager specializing in English job postings. "
            "Your goal is to identify both hard and soft skills in job descriptions. "
            "All decisions must follow the ESCO definitions:\n"
            "• Skill: the ability to apply knowledge and use know-how to complete tasks and solve problems "
            "(e.g. “Python programming”, “data analysis”).\n"
            "• Soft skill: personal, social, or methodological abilities, considered part of ESCO’s ‘attitudes’ "
            "(e.g. “teamwork”, “communication”)."
        ),
        "system_by_prompt": {"ner_old": "You are given a sentence from a job posting. Highlight all the skills, knowledges, and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically."},
        "task_definition": (
            "Task Definition: Annotate both hard skills (technical tools or domain knowledge, e.g. “Python programming”, “AWS”) "
            "and soft skills (behaviors or interpersonal abilities, e.g. “teamwork”, “communication skills”) "
            "by wrapping each mention in @@ and ## within the original sentence. "
        ),
        "demonstrations": (
            "[Retrieved Training Examples]\n{demos}"
        ),
        "background": (
            "[Possible ESCO Occupational Context]\n---\n{general_bg}\n---"
        ),
        "definitions": (
            "[ESCO Skill Phrase Examples — only shown if relevant]\n---\n{skill_defs}\n---"
        ),

        "instruction": {
            "ner": (
                "You are given one sentence from a job posting. "
                "Annotate every skill mention by wrapping it in @@ and ##. "
                "Hard skills are technical tools or domain knowledge (typically noun phrases like software, tools, or fields). "
                "Soft skills are personal, social, or methodological abilities (often verb phrases or adjectives describing capabilities). "
                "If there are no such elements, replicate the sentence identically"
            ),
            "ner_old": (
                "You are given a sentence from a job posting. Highlight all the skills, knowledges, and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically."),
            "ner_cot": (
                "You are given one sentence from a job posting. "
                "Annotate every skill mention by wrapping it in @@ and ##. "
                "Hard skills are technical tools or domain knowledge (typically noun phrases like software, tools, or fields). "
                "Soft skills are personal, social, or methodological abilities (often verb phrases or adjectives describing capabilities). "
                "If no skill is found, repeat the sentence exactly, preserving all punctuation.\n"
                "\nLet’s think step by step. output only the final annotated sentence."
            ),
            "extract": (
                "You are given one sentence from a job posting. "
                "Extract all skill mentions as a JSON array of strings (e.g. [\"Python programming\", \"teamwork\"]). "
                "If no skill is found, return \"None\"."
            )
        }
    },

"chinese_skillspan": {

    "system": (
        "你是一名熟悉中文招聘文本的人力资源与技能本体（ESCO）专家。"
        "你的任务是在原句中标注与能力相关的片段，并严格遵循 ESCO-1.20 的 LKST 平面标注政策：\n"
        "• [L] Language：仅限自然语言能力/证书/等级/使用（如英语六级、日语N2、普通话二甲、能用英语沟通）。\n"
        "• [K] Knowledge：学科/领域/规范/标准/法规/框架/工具体系等“拥有/了解”的知识客体（名词本体）。\n"
        "• [S] Skills：可训练/执行/操作的能力或方法（做事动作或过程，如 制定/开发/调试/部署/评审/配置/优化 等；编程语言与工具使用也归此类）。\n"
        "• [T] Transversal：跨岗位通用能力（沟通、协作、学习、时间管理、抗压、领导力、客户导向等）。\n\n"
        "全局规则：平面标注（不重叠、不嵌套）；采用最小充分片段；并列项逐一切分；剥离“参与/负责/进行/熟悉/掌握/具备/能够/良好/较强”等触发或评价词。"
    ),

    "task_definition": (
        "任务：给定一句中文招聘文本，抽取所有能力相关片段，并在原句内以 NER 形式标注："
        "`@@片段##[L|K|S|T]`。若无可标注内容，原样返回。\n"
        "注意：\n"
        "1) 语言仅指自然语言（如 英语/日语N2/IELTS/CET-6/普通话二甲 等），非自然语言不归 L；\n"
        "2) 编程语言（Python/Java/SQL 等）及工具使用默认归 [S]；\n"
        "3) 名词本体在“知识”语境（了解/熟悉/具备…知识/理解）多判 [K]；处于动作语境（制定/开发/调试/部署/评审）多判 [S]；\n"
        "4) 如遇标签冲突：先按语境判定 K/S；若仍冲突，优先级 L > S > K > T。"
    ),

    "demonstrations_fixed": (
        "【示例 1】\n"
        "输入：能用英语与海外客户邮件沟通，熟练使用Excel进行数据汇总。\n"
        "输出：能用@@英语##[L]与海外客户邮件@@沟通##[S]，熟练使用@@Excel##[S]进行@@数据汇总##[S]\n\n"

        "【示例 2】\n"
        "输入：熟悉药品监管法规，能够制定GMP合规方案。\n"
        "输出：熟悉@@药品监管法规##[K]，能够@@制定GMP合规方案##[S]\n\n"

        "【示例 3】\n"
        "输入：具备良好的沟通能力和时间管理能力，抗压。\n"
        "输出：具备良好的@@沟通能力##[T]和@@时间管理能力##[T]，@@抗压##[T]\n\n"

        "【示例 4】\n"
        "输入：精通Python/Java，掌握SQL和Linux环境。\n"
        "输出：精通@@Python##[S]/@@Java##[S]，掌握@@SQL##[S]和@@Linux 环境##[K]\n\n"

        "【示例 5（K/S 语境判定）】\n"
        "输入：制定视觉流程和通信协议，完成系统调试。\n"
        "输出：@@视觉流程制定##[S]和@@通信协议##[K]，完成@@系统调试##[S]\n\n"

        "【示例 6（名词化术语/编程情境）】\n"
        "输入：负责PLC程序开发与现场调试，了解OPC-UA标准。\n"
        "输出：负责@@PLC 程序开发##[S]与@@现场调试##[S]，了解@@OPC-UA 标准##[K]\n\n"

        "【示例 7（证书/等级归 L）】\n"
        "输入：CET-6，能以英文进行技术汇报。\n"
        "输出：@@CET-6##[L]，能以@@英文##[L]进行@@技术汇报##[S]\n\n"

        "【示例 8（并列切分与最小片段）】\n"
        "输入：统计分析与可视化，熟练PPT。\n"
        "输出：@@统计分析##[S]与@@可视化##[S]，熟练@@PPT##[S]"
    ),

    "demonstrations_fixed_extract": (
        "【EXTRACT 示例 1】\n"
        "输入：能用英语与海外客户邮件沟通，熟练使用Excel进行数据汇总。\n"
        "输出（JSON）：{\n"
        '  "L": ["英语"],\n'
        '  "K": [],\n'
        '  "S": ["沟通", "Excel", "数据汇总"],\n'
        '  "T": []\n'
        "}\n\n"

        "【EXTRACT 示例 2】\n"
        "输入：熟悉药品监管法规，能够制定GMP合规方案。\n"
        "输出（JSON）：{\n"
        '  "L": [],\n'
        '  "K": ["药品监管法规"],\n'
        '  "S": ["制定GMP合规方案"],\n'
        '  "T": []\n'
        "}\n\n"

        "【EXTRACT 示例 3】\n"
        "输入：具备良好的沟通能力和时间管理能力，抗压。\n"
        "输出（JSON）：{\n"
        '  "L": [],\n'
        '  "K": [],\n'
        '  "S": [],\n'
        '  "T": ["沟通能力", "时间管理能力", "抗压"]\n'
        "}\n\n"

        "【EXTRACT 示例 4】\n"
        "输入：精通Python/Java，掌握SQL和Linux环境。\n"
        "输出（JSON）：{\n"
        '  "L": [],\n'
        '  "K": ["Linux 环境"],\n'
        '  "S": ["Python", "Java", "SQL"],\n'
        '  "T": []\n'
        "}\n\n"

        "【EXTRACT 示例 5（K/S 语境判定）】\n"
        "输入：制定视觉流程和通信协议，完成系统调试。\n"
        "输出（JSON）：{\n"
        '  "L": [],\n'
        '  "K": ["通信协议"],\n'
        '  "S": ["视觉流程制定", "系统调试"],\n'
        '  "T": []\n'
        "}\n\n"

        "【EXTRACT 示例 6（证书/等级归 L）】\n"
        "输入：CET-6，能以英文进行技术汇报。\n"
        "输出（JSON）：{\n"
        '  "L": ["CET-6", "英文"],\n'
        '  "K": [],\n'
        '  "S": ["技术汇报"],\n'
        '  "T": []\n'
        "}"
    ),

    "demonstrations": "[Retrieved Training Examples]\n{demos}",
    "background": "[Possible ESCO Occupational Context]\n---\n{general_bg}\n---",
    "definitions": "[ESCO Skill Phrase Examples — only shown if relevant]\n---\n{skill_defs}\n---",

    "instruction": {
        "ner": (
            "你将获得一句中文招聘文本。请在原句中用 `@@片段##[L|K|S|T]` 标注所有能力相关片段，遵循：\n"
            "1) 平面标注：不重叠、不嵌套；并列项逐一切分；\n"
            "2) 最小充分片段：去掉轻动词/评价/程度与连接词，仅保留能自解释的核心片段；\n"
            "3) K/S 语境：名词本体处于“知识/了解/熟悉/具备…知识/理解”语境多判 K；在“制定/设计/评审/配置/开发/调试/部署/维护/优化/调优”等动作语境多判 S；\n"
            "4) 语言 L：仅自然语言/证书/等级/使用；编程语言与工具使用归 S；\n"
            "5) 冲突兜底：先按语境判定 K/S；若仍冲突，则按优先级 L > S > K > T；\n"
            "6) 无能力相关内容时原样返回。\n"
            "输出仅包含一行：标注后的句子。"
        ),
        "ner_cot": (
            "你将获得一句中文招聘文本。先在心中逐步思考边界与标签决策，但**只输出最终标注**：\n"
            "在原句中用 `@@片段##[L|K|S|T]` 标注全部能力相关片段；不重叠、不嵌套；最小充分片段；\n"
            "K/S 先看语境；编程语言与工具使用归 S；语言仅自然语言；若仍冲突，优先级 L > S > K > T；\n"
            "无可标注则原样返回。\n"
            "注意：仅输出标注后的句子，不要解释过程。"
        ),
        "extract": (
            "你将获得一句中文招聘文本。请**抽取**其中所有能力相关项，按 ESCO-1.20 四维度输出为 JSON：\n"
            '{\n  "L": [...],  // Language（仅自然语言/等级/证书/使用）\n'
            '  "K": [...],  // Knowledge（学科/领域/规范/标准/法规/框架/对象等名词本体）\n'
            '  "S": [...],  // Skills（可执行/操作/训练的方法与技能；含编程语言与工具的实际使用）\n'
            '  "T": [...]   // Transversal（沟通/协作/学习/时间管理/抗压/领导力等通用能力）\n}\n'
            "标注规则：\n"
            "• 不重叠：同一片段只放入一个最合适维度；\n"
            "• 最小充分片段：剥离触发/评价/程度/连接词，仅保留可自解释核心；\n"
            "• 先按语境区分 K/S；若仍冲突，优先级 L > S > K > T；\n"
            "• 语言严格限定为自然语言；编程语言与工具实际使用统一归 S；\n"
            "• 去重且保留原文用词、大小写与全/半角；\n"
            "• 仅输出**合法 JSON**，不要任何额外解释或标点。\n\n"
            "现在开始：仅输出 JSON。"
        )
    }
}
,
    "fijo": {
        "system": (
            "You are an expert human-resource manager specializing in analyzing French job advertisements, "
            "particularly in the insurance sector and other relevant domains. "
            "Your goal is to identify all skills (hard and soft) required from the candidate in the job description.\n"
            "Skills are annotated according to the FIJO definitions based on the public AQESSS repositories and proprietary skill sets provided by collaborators:\n"
            "• Thoughts (PENSEE): intellectual abilities, analysis, and problem-solving skills.\n"
            "• Results (RESULTATS): goal orientation, efficiency, performance management.\n"
            "• Relational (RELATIONNEL): communication, teamwork, negotiation, leadership.\n"
            "• Personal (PERSONNEL): personal qualities, autonomy, adaptability.\n"
            "Include both technical abilities (hard skills) and behavioral abilities (soft skills)."
        ),
        "system_by_prompt": {
            "ner_old": "You are an expert human resource manager in the insurance industry in France. You need to analyse skills required in job offers.",
        },
        "task_definition": (
            """Task Definition: For a single tokenized sentence (tokens are space-separated), annotate every skill/competency mention by surrounding its complete yet concise phrase with @@ and ##.
                Constraints:
                • Keep tokens and their order unchanged; only insert @@ and ## (non-overlapping spans).
                • Include necessary function words that form a conventional ability phrase (e.g., maîtrise de …, gestion de projet, résolution de problèmes, service client).
                • Split coordinated skills into separate spans (e.g., communication et négociation → two spans).
                • Exclude qualifications (degrees/certifications), time-based experience, job titles, industries/domains, and environment descriptors.
                • If no skill is present, reproduce the sentence exactly.
                Output: Only the final annotated sentence with @@ and ##, nothing else."""
        ),

        "background": (
            "[FIJO RAG Skill Lexicon (CSV-derived skills list) • Retrieved if Highly Similar]\n---\n{general_bg}\n---"
        ),

        "definitions": (
            "[FIJO RAG Competency/Skill Definitions (Descripteur level) • Retrieved if Relevant]\n---\n{skill_defs}\n---"
        ),

        "demonstrations": (
            "[In-Domain Few-Shot & Retrieved Annotations]\n{demos}"
        ),
        "demonstrations_fixed": (
            """Negative examples (should not be annotated):
            Input: "Licence en informatique et 5 ans d'expérience exigés ."
            Output: "Licence en informatique et 5 ans d'expérience exigés ." # Education/years of experience are not annotated
            Input: "Chef de projet ( assurance ) au sein d'une équipe dynamique ."
            Output: "Chef de projet ( assurance ) au sein d'une équipe dynamique ." # Position/industry/environment are not annotated
            Boundaries (avoid being too long/too short)
            Input: "Orienté résultats et sens du service client ."
            Output: "@@Orienté résultats## et @@sens du service client## ." # Two phrases, each forming a span"""
        ),
        "instruction": {
            "ner": (
                """You are given a sentence from a French job description. Highlight all required skills/competencies by enclosing each in @@ and ##. Keep tokens and spacing unchanged; only insert markers. Use concise spans that preserve conventional phrases; split coordinated skills into separate spans. Do not include degrees/certifications, time-based experience, job titles, industries/domains, or environment descriptors. If none, reproduce the sentence. Output only the final annotated sentence"""
            ),
            "ner_old": (
                "You are given a sentence from an insurance job description in French. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically."),
            "ner_cot": (
                """You are given a sentence from a French job description. Highlight all required skills/competencies by enclosing each in @@ and ##. Keep tokens and spacing unchanged; only insert markers. Use concise spans that preserve conventional phrases; split coordinated skills into separate spans. Do not include degrees/certifications, time-based experience, job titles, industries/domains, or environment descriptors. If none, reproduce the sentence. Output only the final annotated sentence"""
                "Think step by step silently but do not reveal your reasoning; output only the final annotated sentence."
            ),

            "extract": (
                "You are given a sentence from a French job description. "
                "Extract all candidate-required skills as a list (one per line). "
                "Include both technical and behavioral abilities corresponding to the categories Thoughts, Results, Relational, and Personal. "
                "Do NOT include qualifications, time-based experience, job titles, industries/domains, or company/team/environment descriptors. "
                "If no skill is found, return \"None\"."
            )
        }
    },
    "gnehm": {
        "system": (
            "You are an expert annotator for German ICT job postings. Label only ICT skills (technologies, tools, programming languages, frameworks, libraries, methodologies). Do not label job titles, company names, gender markers (m/w/d), education, years of experience, or language proficiency, unless the gold definition requires them. Aim for complete spans of the skill mention."
        ),
        "system_by_prompt": {
            "ner_old": "You are an expert human resource manager in information and communication technology (ICT) from Germany. You need to analyse skills required in German job offers."
        },
        "task_definition": (
            """Annotate all ICT skill mentions in a German ICT job-ad sentence. Label only technologies, tools, programming languages, frameworks, libraries, and methodologies. Do NOT label job titles, company names, gender markers (m/w/d), education, years of experience, or language proficiency.
        Tokens are space-separated; keep tokens and their order unchanged, only insert markers.
        Output format: surround each ICT skill span with @@ and ## (non-overlapping), capturing the complete skill phrase (e.g., “Microsoft Azure DevOps Pipelines”, “unit testing”).
        If no ICT skill is present, return the sentence unchanged.
        Output only the final annotated sentence with @@ and ##, nothing else."""
                ),
        "background": (
            "[ICT Occupational Context – Retrieved If Relevant]\n---\n{general_bg}\n---"
        ),

        "definitions": (
            "[Skill Requirement Definitions – Retrieved If Relevant]\n---\n{skill_defs}\n---"
        ),

        "demonstrations": (
            "[In-Domain Few-Shot & Retrieved Annotations]\n{demos}"
        ),
        "demonstrations_fixed": (
        """Input: "Erfahrung mit Java und Docker erforderlich ."
        Output: "Erfahrung mit @@Java## und @@Docker## erforderlich ."
        Input: "Full-Stack Software Developer ( m / w / d )"
        Output: "Full-Stack Software Developer ( m / w / d )"  # 岗位名不标
        Input: "Kenntnisse in Microsoft Azure DevOps Pipelines ."
        Output: "Kenntnisse in @@Microsoft Azure DevOps Pipelines## ."
        Input: "Fließende Deutschkenntnisse , Bachelor in Informatik ."
        Output: "Fließende Deutschkenntnisse , Bachelor in Informatik ."""  # 语言/学历不标
        ),
        "instruction": {
            "ner": (
                """You are given a tokenized German sentence from an ICT job ad (tokens are space-separated).
                    Surround every ICT skill span with @@ and ## (no extra text).
                    Constraints:
                    Output a single line: the original tokens with only @@ and ## inserted.
                    Keep token order unchanged; do not add/remove tokens.
                    Spans are non-overlapping; capture the whole skill phrase (e.g., “Microsoft Azure”, “unit testing”, “CI/CD”).
                    Do not label job titles (“Engineer”, “Manager”, …), company names, (m / w / d), education, years of experience, or languages.
                    If no skill is present, return the sentence unchanged."""
            ),
            "ner_old": ("You are given an extract from a job advertisement in German. Highlight all the IT/Technology skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically."
            ),
            "ner_cot": (
                """You are given a tokenized German sentence from an ICT job ad (tokens are space-separated).
                   Surround every ICT skill span with @@ and ## (no extra text).
                   Constraints:
                   Output a single line: the original tokens with only @@ and ## inserted.
                   Keep token order unchanged; do not add/remove tokens.
                   Spans are non-overlapping; capture the whole skill phrase (e.g., “Microsoft Azure”, “unit testing”, “CI/CD”).
                   Do not label job titles (“Engineer”, “Manager”, …), company names, (m / w / d), education, years of experience, or languages.
                   If no skill is present, return the sentence unchanged."""
                "Think step by step silently but do not reveal your reasoning; output only the final annotated sentence"
            ),
            "extract": (
                "You are given a sentence from a German job description. "
                "Extract all explicit requirements as a JSON array of strings "
                "(e.g. [\"Bachelor in Informatik\", \"5 Jahre Erfahrung in Vertrieb\", \"Fließende Englischkenntnisse\"]). "
                "If no requirement is found, return \"None\"."
            )
        }
    },
    "green": {
        "system": (
          """You are an expert annotator for English job advertisements.
            Label only candidate-required skills/competencies (both hard and soft skills).
            Do NOT label degrees/certifications, time-based experience or seniority terms, job titles/roles, industries/domains, company/team/environment descriptors, or locations.
            Follow the benchmark style and keep spans precise yet complete."""
        ),
        "system_by_prompt": {
            "ner_old": "You are an expert human resource manager. You need to analyse skills required in job offers.",
        },
        "task_definition": (
          """Task Definition (Sentence-level, tokens are space-separated):
            Insert @@ and ## around each required skill span in the original sentence.
            Constraints
            • Keep tokens and their order unchanged; only insert the markers (non-overlapping).
            • Include necessary function words to form a conventional skill phrase (e.g., project management, stakeholder management, attention to detail, problem solving, customer service).
            • Split coordinated skills into separate spans (e.g., “communication and negotiation” → two spans).
            • Exclude degrees/certifications, time-based experience, seniority words, job titles/roles, industries/domains, company/team/environment descriptors.
            • If no skill is present, reproduce the sentence exactly.
            Output: Only the final annotated sentence with @@ and ##, nothing else."""
        ),

        "background": (
          "[Optional Domain Hints – Use only if highly similar]\n---\n{general_bg}\n---"
        ),
        "definitions": (
            "[ESCO Skill Phrase Examples – Retrieved If Relevant]\n---\n{skill_defs}\n---"
        ),
        "demonstrations": (
            "[In‑Domain Few‑Shot & Retrieved Annotations]\n{demos}"
        ),
"demonstrations_fixed": (
            """Input: "You must be proactive and have good time management ."
            Output: "You must be @@proactive## and have @@good time management## ."
            
            Input: "Ability to work under pressure and strong problem solving skills ."
            Output: "Ability to @@work under pressure## and @@strong problem solving skills## ."
                  
            Input: "Excellent communication and negotiation with stakeholders ."
            Output: "Excellent @@communication## and @@negotiation## with stakeholders ."
            
            Input: "Proficiency in project management and stakeholder management ."
            Output: "Proficiency in @@project management## and @@stakeholder management## ."""
        ),

            "instruction": {
          "ner": (
            """
            You are given one sentence from a job description. Highlight all and only the candidate’s required skills/competencies by enclosing each skill phrase in @@ and ##. Keep tokens/spaces/ punctuation unchanged; only insert markers. Use concise spans that preserve the conventional phrase; split coordinated skills into separate spans. Do not include degrees/certifications, experience requirements or seniority, job titles, industries/domains, or environment descriptors. If none, reproduce the sentence exactly. Output only the annotated sentence.
            """
          ),
        "ner_old": (
       "You are given a sentence from a job description in various fields like IT, finance, healthcare, and sales. Highlight all the skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'.  If there are no such element in the sentence, replicate the sentence identically."
                  ),

          "ner_cot": (
            "You are given one sentence from a job description. "
            "Highlight all and only the candidate’s required skills/competencies by enclosing each skill phrase in @@ and ##. "
            "Keep tokens/spaces/punctuation unchanged; only insert markers. "
            "Use concise spans that preserve the conventional phrase; split coordinated skills into separate spans. "
            "Do not include degrees/certifications, experience requirements or seniority, job titles, industries/domains, or environment descriptors. "
            "If none, reproduce the sentence exactly. Output only the annotated sentence. "
            "Think step by step silently but do not reveal your reasoning; output only the annotated sentence."
        ),
                  "extract": (
            "You are given one sentence from a job description. "
            "Extract all candidate-required skills as a list, one per line. "
            "Do NOT include qualifications, experience requirements, job titles, domains, or environment descriptions. "
            "If no skill is found, return \"None\".\n\n"
            "Think briefly but output only the list."
          )
        },
    },
    "sayfullina": {
                "system": (
                    "You are an expert human-resource manager specializing in analyzing job postings. "
                    "Your goal is to identify only soft skills in job descriptions. "
                    "All decisions must follow the 2018 Collins Dictionary definition:\n"
                    "• Soft skills are desirable qualities for certain forms of employment that do not depend on acquired knowledge; "
                    "they include common sense, the ability to deal with people, and a positive flexible attitude.\n"
                    "• Only annotate skills that describe the candidate’s personal, social, or methodological abilities, "
                    "not company culture or environment."
                ),
                "system_by_prompt": {
                    "ner_old": "You are an expert human resource manager. You need to detect and analyse soft skills required in job offeres"
                },
                "task_definition": (
                    "Task Definition: Annotate only soft skills (behaviors, interpersonal abilities, or methodological traits) "
                    "by wrapping each mention in @@ and ## within the original sentence. "
                    "If no candidate soft skill mention is present, repeat the sentence exactly, preserving all punctuation."
                ),
                "background": (
                    "[Soft Skill Context – Retrieved If Highly Similar]\n---\n{general_bg}\n---"
                ),
                "definitions": (
                    "[Soft Skill Phrase Examples – Retrieved If Relevant]\n---\n{skill_defs}\n---"
                ),
                "demonstrations": (
                    "[In-Domain Few-Shot & Retrieved Annotations]\n{demos}"
                ),
                "instruction": {
                    "ner": (
                        "You are given a sentence from a job posting. "
                        "Annotate every soft skill mention by wrapping it in @@ and ##. "
                        "Soft skills are personal, social, or methodological abilities "
                        "(e.g. @@proactive##, @@good time management##). "
                        "Do not annotate skills that describe the company, team, or work environment. "
                        "If no candidate soft skill is found, repeat the sentence exactly, preserving all punctuation."
                    ),
                    "ner_old": (
                        "You are given a sentence from a job advertisement. Highlight all the soft skills and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically."
                    ),
                    "ner_cot": (
                        "You are given a sentence from a job posting. "
                        "Annotate every soft skill mention by wrapping it in @@ and ##. "
                        "Soft skills are personal, social, or methodological abilities "
                        "(e.g. @@proactive##, @@good time management##). "
                        "Do not annotate skills that describe the company, team, or work environment. "
                        "If no candidate soft skill is found, repeat the sentence exactly, preserving all punctuation.\n\n"
                        "Let’s think step by step,but DO NOT reveal your reasoning; only output the marked sentence."
                    ),
                    "extract": (
                        "You are given a sentence from a job posting. "
                        "Extract all soft skill mentions as a JSON array of strings (e.g. [\"proactive\", \"good time management\"]). "
                        "Do not include skills describing the company or team. "
                        "If no candidate soft skill is found, return \"None\"."
                    )
                }
            },
    "kompetencer": {
            "system": (
            "You are an expert human‑resource manager specializing in Danish job postings. "
            "Your goal is to identify both hard and soft skills in job descriptions. "
            "All decisions must follow the ESCO definitions:\n"
            "• Skill: the ability to apply knowledge and use know‑how to complete tasks and solve problems "
            "(e.g. “Python programming”, “data analysis”).\n"
            "• Soft skill: personal, social, or methodological abilities, considered part of ESCO’s ‘attitudes’ "
            "(e.g. “teamwork”, “communication”)."
        ),
        "system_by_prompt": {
                        "ner_old": "You are an expert human resource manager. You need to analyse skills required in job offers"
                    },
        "task_definition": (
            "Task Definition: Annotate both hard skills (technical tools or domain knowledge, e.g. “Python programming”, “AWS”) "
            "and soft skills (behaviors or interpersonal abilities, e.g. “teamwork”, “communication skills”) "
            "by wrapping each mention in @@ and ## within the original sentence. "
            "If no skill mention is present, repeat the sentence exactly, preserving all original punctuation."
        ),
        "background": (
            "[ESCO Occupational Context – Retrieved If Highly Similar]\n---\n{general_bg}\n---"
        ),
        "definitions": (
            "[ESCO Skill Phrase Examples – Retrieved If Relevant]\n---\n{skill_defs}\n---"
        ),
        "demonstrations": (
            "[In‑Domain Few‑Shot & Retrieved Annotations]\n{demos}"
        ),
        "instruction": {
            "ner": (
                "You are given a sentence from a job posting in Danish. "
                "Annotate every skill mention by wrapping it in @@ and ##. "
                "Hard skills are technical tools or domain knowledge (typically noun phrases like software, tools, or fields). "
                "Soft skills are personal, social, or methodological abilities (often verb phrases or adjectives describing capabilities). "
                "If no skill is found, repeat the sentence exactly, preserving all punctuation."
            ),
            "ner_old": (
                "You are given a sentence from a job description in Danish. Highlight all the skills, knowledges, and competencies that are required from the candidate, by surrounding them with tags '@@' and '##'. If there are no such element in the sentence, replicate the sentence identically."
            ),
            "ner_cot": (
                "You are given a sentence from a job posting in Danish. "
                "Annotate every skill mention by wrapping it in @@ and ##. "
                "Hard skills are technical tools or domain knowledge (typically noun phrases like software, tools, or fields). "
                "Soft skills are personal, social, or methodological abilities (often verb phrases or adjectives describing capabilities). "
                "If no skill is found, repeat the sentence exactly, preserving all punctuation.\n\n"
                "Let’s think step by step,but DO NOT reveal your reasoning; only output the marked sentence."
            ),
            "extract": (
                "You are given a sentence from a job posting. "
                "Extract all skill mentions as a JSON array of strings (e.g. [\"Python programming\", \"teamwork\"]). "
                "If no skill is found, return \"None\"."
            )
            }
    }
    }