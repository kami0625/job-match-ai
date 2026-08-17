"""RAG 求职匹配业务逻辑层

职责：文档入库、混合检索问答、NL2SQL 岗位查询、简历匹配评分、匹配图表生成。

分层约束（见 00_项目开发规范与工程约束.md）：
- 本层只负责业务逻辑编排；
- 底层能力（大模型 / 向量库 / MySQL / 文档解析）全部复用 app/utils 与 app/dao 层；
- 禁止直接 requests 调用 11434 端口，所有大模型能力统一走 OllamaClient。
"""

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

from app.config import (
    RAG_BM25_TOP_K,
    RAG_TOP_K,
    RAG_VECTOR_TOP_K,
    UPLOAD_DIR,
)
from app.dao.chroma_db import ChromaDB
from app.dao.mysql_db import MySQLDB
from app.utils.common_tools import (
    AppError,
    extract_json_from_text,
    generate_id,
    get_logger,
    now_str,
    SQL_PCT_PLACEHOLDER,
)
from app.utils.doc_parser import parse_document, split_text
from app.utils.ollama_client import OllamaClient

logger = get_logger("rag_service")

# job_info 表结构描述（供 NL2SQL 提示词使用）
JOB_TABLE_SCHEMA = """
表 job_info（岗位信息表）字段说明：
- id: INT 岗位ID（主键）
- job_name: VARCHAR 岗位名称，如 "Java开发工程师"
- company: VARCHAR 公司名称
- city: VARCHAR 所在城市，如 "上海"
- salary_min: INT 薪资下限（单位 K），如 15 表示 15K
- salary_max: INT 薪资上限（单位 K）
- skill_require: TEXT 技能要求
- education: VARCHAR 学历要求，如 "本科"
- experience: VARCHAR 经验要求，如 "3-5年"
- job_desc: TEXT 岗位描述
- create_time: DATETIME 录入时间
"""

# 常见技术栈关键词表（规则评分兜底 / 技能提取用）
SKILL_KEYWORDS = [
    "Java", "Python", "Go", "C++", "JavaScript", "TypeScript", "Vue", "React",
    "Spring", "SpringBoot", "MyBatis", "MySQL", "Redis", "MongoDB", "Elasticsearch",
    "Kafka", "RocketMQ", "Docker", "Kubernetes", "Linux", "Nginx", "Git",
    "Hadoop", "Spark", "Flink", "Hive", "机器学习", "深度学习", "TensorFlow", "PyTorch",
    "NLP", "计算机视觉", "数据分析", "SQL", "算法", "数据结构", "微服务", "分布式",
    "消息队列", "高并发", "性能优化", "JVM", "多线程", "网络编程", "自动化测试", "CI/CD",
    "云计算", "Android", "iOS", "Flutter", "小程序", "前端", "后端", "全栈",
]

class BM25Index:
    """轻量 BM25 索引（rank_bm25 + jieba 中文分词）"""

    def __init__(self) -> None:
        self._documents: list = []
        self._bm25: Optional[BM25Okapi] = None

    def rebuild(self, documents: list) -> None:
        """基于文档列表重建索引"""
        tokenized = [list(jieba.cut(d)) for d in documents]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._documents = documents

    def search(self, query: str, top_k: int) -> list:
        """关键词检索，返回 [{"index","document","score"}, ...]"""
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(list(jieba.cut(query)))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            {"index": idx, "document": self._documents[idx], "score": float(score)}
            for idx, score in ranked[:top_k]
            if score > 0
        ]

class RagService:
    """RAG 求职匹配业务服务（单例）"""

    _instance: Optional["RagService"] = None

    def __new__(cls) -> "RagService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.chroma = ChromaDB()
        self.mysql = MySQLDB()
        self._bm25 = BM25Index()
        self._bm25_doc_count: int = -1
        self._bm25_metas: list = []

    # ==================== 1. 文档入库 ====================

    def upload_document(self, file_bytes: bytes, filename: str) -> dict:
        """上传文档并入库（自动解析 -> 分块 -> 向量化 -> 写入 Chroma）

        去重策略（防止向量库膨胀）：
        - 以文件内容 SHA256 生成稳定 doc_id；
        - 同一内容重复上传 → 先删旧块再写入（覆盖更新，返回 updated=True）；
        - 不同内容 → 生成新 doc_id，增量新增，不影响全库。

        Args:
            file_bytes: 文件二进制内容
            filename: 原始文件名（用于识别扩展名）
        Returns:
            {"doc_id", "file_name", "chunk_count", "create_time", "updated"}
        """
        ext = Path(filename).suffix.lower()
        if ext not in (".pdf", ".docx"):
            raise AppError("仅支持上传 PDF / DOCX 格式文件", code=400)
        save_path = UPLOAD_DIR / f"{generate_id('doc')}{ext}"
        save_path.write_bytes(file_bytes)

        parsed = parse_document(str(save_path))
        chunks = split_text(parsed["text"])
        if not chunks:
            raise AppError("文档解析后无有效文本内容", code=400)

        # 内容哈希生成稳定 doc_id，实现重复入库覆盖更新
        content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        doc_id = f"doc_{content_hash}"
        updated = self.chroma.exists_by_doc_id(doc_id)
        if updated:
            logger.info("检测到文档重复入库，执行覆盖更新 doc_id=%s", doc_id)
            self.chroma.delete_by_doc_id(doc_id)

        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "file_name": filename,
                "chunk_id": i,
                "create_time": now_str(),
            }
            for i in range(len(chunks))
        ]
        self.chroma.add_documents(ids=ids, documents=chunks, metadatas=metadatas)
        self._bm25_doc_count = -1  # 标记 BM25 索引失效，下次检索自动重建
        logger.info("文档入库成功 doc_id=%s chunks=%d updated=%s", doc_id, len(chunks), updated)
        return {
            "doc_id": doc_id,
            "file_name": filename,
            "chunk_count": len(chunks),
            "create_time": now_str(),
            "updated": updated,
        }

    # ==================== 2. RAG 智能问答 ====================

    def chat(self, query: str, top_k: Optional[int] = None) -> dict:
        """RAG 智能问答：查询改写 -> 双路召回 -> 合并精排 -> 大模型生成

        Args:
            query: 用户问题
            top_k: 返回引用片段数（默认 RAG_TOP_K）
        Returns:
            {"answer", "sources": [{"file_name","content","score"}], "query_rewritten"}
        """
        k = min(top_k or RAG_TOP_K, 10)

        # 1. 查询改写：扩写模糊提问，提升召回（失败回退原文）
        rewrite_failed = False
        try:
            rewritten = self.ollama.chat(
                "把下面问题改写为更清晰的检索问句,只输出改写结果一句:" + query,
                system="你是查询改写助手,只输出一句改写结果,不要解释。",
                temperature=0.1,
            ).strip()
            if not rewritten:
                rewritten = query
                rewrite_failed = True
        except Exception:
            rewritten = query
            rewrite_failed = True

        # 2. 双路召回（向量语义 + BM25 关键词），合并去重
        candidates = self._recall(rewritten, top_k=max(k * 2, 6))
        if not candidates:
            logger.info("知识库无可用内容，直接问答 query=%s", query[:50])
            answer = self.ollama.chat(
                query,
                system="你是一个专业的求职行业知识助手，请清晰、简洁地回答问题。如果问题不在你的知识范围内,请如实说明。",
            )
            return {"answer": answer, "sources": [], "query_rewritten": rewritten, "rewrite_failed": rewrite_failed}

        # 3. 精排：按得分降序取 Top K
        selected = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:k]
        sources = [
            {
                "file_name": c.get("file_name", ""),
                "content": c.get("document", ""),
                "score": round(c.get("score", 0), 4),
            }
            for c in selected
        ]

        # 4. 拼接上下文，生成带引用的回答
        context = "\n\n".join(f"[资料{i+1}]\n{s['content']}" for i, s in enumerate(sources))
        system_prompt = (
            "你是基于参考资料的中文求职行业助手。回答用中文,严格基于下面给出的参考资料回答。"
            "规则:1. 参考资料足以回答时,直接给出答案并可在末尾标注依据[资料N];"
            "2. 参考资料不足以回答问题时,只回复一句「知识库中未找到相关资料,无法回答该问题」,禁止编造或展开任何内容。"
        )
        user_prompt = "参考资料:" + context[:3000] + "\n\n问题:" + query
        try:
            answer = self.ollama.chat(user_prompt, system=system_prompt, temperature=0.3)
        except Exception as exc:
            logger.warning("RAG 生成答案失败,降级直接问答: %s", exc)
            answer = self.ollama.chat(
                query,
                system="你是基于参考资料的中文求职行业助手,请清晰、简洁地回答问题。",
            )
        return {"answer": answer, "sources": sources, "query_rewritten": rewritten, "rewrite_failed": rewrite_failed}

    def _recall(self, query: str, top_k: int = 6) -> list:
        """双路召回：向量语义检索 + BM25 关键词检索，按相似度合并去重

        Returns:
            [{"document","file_name","score"}, ...] 按 score 降序
        """
        merged = {}

        # 路径一：向量语义检索（Chroma）
        try:
            vec_items = self.chroma.semantic_search(query, top_k=top_k)
            for item in vec_items:
                # distance 越小越相似 -> score = 1 - distance（clip 到 0~1）
                dist = item.get("distance")
                score = max(0.0, 1.0 - dist) if dist is not None else 0.0
                merged[item["id"]] = {
                    "document": item.get("document", ""),
                    "file_name": (item.get("metadata") or {}).get("file_name", ""),
                    "score": score,
                    "_rank": 0.5,
                }
        except Exception as exc:
            logger.warning("向量召回失败,仅用关键词: %s", exc)

        # 路径二：BM25 关键词检索
        try:
            if self._bm25_index_ready():
                bm25_items = self._bm25.search(query, top_k=top_k)
                for item in bm25_items:
                    key = f"bm25_{item['index']}"
                    score = min(item.get("score", 0), 5.0) / 5.0  # 归一化
                    if key in merged:
                        merged[key]["score"] = max(merged[key]["score"], score)
                        merged[key]["_rank"] += 0.3
                    else:
                        merged[key] = {
                            "document": item.get("document", ""),
                            "file_name": self._bm25_metas[item["index"]].get("file_name", "")
                            if item["index"] < len(self._bm25_metas) else "",
                            "score": score,
                            "_rank": 0.3,
                        }
        except Exception as exc:
            logger.warning("BM25 召回失败: %s", exc)

        # 合并打分：score = 0.7*相似度 + 0.3*召回rank加成
        results = []
        for key, item in merged.items():
            final = round(item["score"] * 0.7 + item["_rank"] * 0.3, 4)
            results.append({
                "document": item["document"],
                "file_name": item["file_name"],
                "score": final,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _bm25_index_ready(self) -> bool:
        """确保 BM25 索引已构建（基于 Chroma 当前文档数懒加载）"""
        try:
            count = self.chroma.count()
        except Exception:
            count = 0
        if self._bm25_doc_count != count:
            docs, metas = [], []
            try:
                coll = self.chroma.get_collection()
                got = coll.get(include=["documents", "metadatas"])
                for i in range(len(got.get("ids") or [])):
                    docs.append((got["documents"] or [""])[i] or "")
                    metas.append((got["metadatas"] or [{}])[i] or {})
            except Exception as exc:
                logger.warning("BM25 索引重建失败: %s", exc)
            self._bm25.rebuild(docs)
            self._bm25_metas = metas
            self._bm25_doc_count = count
        return self._bm25 is not None and self._bm25_metas is not None

    def parse_and_import_pasted_jd(self, jd_text: str) -> dict:
        """批量粘贴 JD 入库：LLM 逐段解析为结构化字段写入 job_info（去重）"""
        segments = [s.strip() for s in re.split(r"\n---\n|---", jd_text) if s.strip()]
        if not segments:
            raise AppError("JD 文本为空,请粘贴岗位描述", code=400)

        system_prompt = (
            "你是国内招聘 JD 解析专家。给定一段中文招聘 JD 文本,提取结构化字段。请严格输出合法 JSON(不要任何其他内容、不要 markdown 包裹、不要多余解释)。JSON 字段:job_name(岗位名称,必填), company(公司名,必填), city(城市,如『上海』『杭州』;未提则填『未知』), salary_min(数字 K/月,无则 0), salary_max(数字 K/月,无则 0), skill_require(技能要求,逗号分隔), education(学历要求,未提则『不限』), experience(经验要求,未提则『不限』), job_desc(完整岗位描述,不超过 500 字)。薪资规则:15-25K 转 salary_min=15,salary_max=25;15K 以上转 salary_min=15,salary_max=0;面议则两个都填 0。字符串内的双引号用中文引号『』替代。"
        )

        parsed_count = 0
        inserted = 0
        skipped = 0
        errors = []
        details = []

        for i, seg in enumerate(segments):
            try:
                raw = self.ollama.chat(seg[:2000], system=system_prompt, temperature=0.1)
                data = extract_json_from_text(raw)
                if not isinstance(data, dict):
                    raise AppError("LLM 未返回有效 JSON")
                job_name = str(data.get("job_name") or "").strip()
                company = str(data.get("company") or "").strip()
                if not job_name or not company:
                    raise AppError("必填字段缺失 job_name=" + repr(job_name) + " company=" + repr(company))
                city = str(data.get("city") or "未知").strip()
                s_min = int(data.get("salary_min") or 0)
                s_max = int(data.get("salary_max") or 0)
                skill_require = str(data.get("skill_require") or "").strip()
                education = str(data.get("education") or "不限").strip()
                experience = str(data.get("experience") or "不限").strip()
                job_desc = str(data.get("job_desc") or seg)[:500]
                exists = self.mysql.query_one(
                    "SELECT id FROM job_info WHERE data_source = 'user_paste' "
                    "AND job_name = %s AND company = %s AND city = %s",
                    (job_name, company, city),
                )
                if exists:
                    skipped += 1
                    details.append({"status": "skipped", "job_name": job_name})
                    continue
                self.mysql.insert_one(
                    "INSERT INTO job_info (job_name, company, city, salary_min, salary_max, "
                    "skill_require, education, experience, job_desc, data_source, external_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'user_paste',%s)",
                    (job_name, company, city, s_min, s_max, skill_require,
                     education, experience, job_desc, "paste_" + generate_id("id")),
                )
                inserted += 1
                details.append({"status": "inserted", "job_name": job_name, "company": company, "city": city})
                parsed_count += 1
            except Exception as exc:
                skipped += 1
                errors.append("第" + str(i + 1) + "段: " + str(exc)[:200])

        return {
            "parsed": parsed_count,
            "inserted": inserted,
            "skipped": skipped,
            "errors": errors[:10],
            "details": details,
        }

    def get_job(self, job_id: int) -> dict:
        """岗位详情"""
        job = self.mysql.query_one("SELECT * FROM job_info WHERE id = %s", (job_id,))
        if not job:
            raise AppError(f"岗位 ID={job_id} 不存在", code=404)
        return job

    def list_jobs(self, limit=20, offset=0):
        limit = min(max(int(limit), 1), 200)
        offset = max(int(offset), 0)
        rows = self.mysql.query(
            "SELECT id, job_name, company, city, salary_min, salary_max, "
            "education, experience, create_time FROM job_info "
            "ORDER BY create_time DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        total = self.mysql.query_one("SELECT COUNT(*) AS cnt FROM job_info") or {"cnt": 0}
        return {"items": rows, "total": total.get("cnt", 0), "limit": limit, "offset": offset}

    # ============ NL2SQL ============

    def query_jobs(self, nl_query, limit=20):
        sql = self._generate_sql(nl_query, limit)
        exec_sql = sql.replace("%", SQL_PCT_PLACEHOLDER)
        result = self.mysql.query(exec_sql)
        return {"sql": sql, "result": result, "count": len(result)}

    def _generate_sql(self, nl_query, limit):
        system_prompt = (
            "你是 MySQL 专家,把自然语言查询转为 SQL。"
            + JOB_TABLE_SCHEMA
            + "约束:1. 只生成 SELECT;2. 只查 job_info 表;3. 字符串加单引号,数值直接比较;"
            "4. 多条件用 AND;5. 薪资语义:'X K 以上/起/超过 X K' 转 salary_max > X(严格大于);"
            "'X K 及以上/至少 X K' 转 salary_max >= X;'X K 以下' 转 salary_max < X;"
            "'X K 到 Y K / X-Y K' 转 salary_max >= X AND salary_min <= Y;"
            "6. 城市/岗位名/技能用 LIKE 模糊匹配;7. ORDER BY create_time DESC LIMIT {limit};"
            "8. 只输出 SQL 语句,不要任何解释/注释/markdown 标记"
        ).format(limit=limit)
        sql = self.ollama.chat(nl_query, system=system_prompt, temperature=0.1).strip()
        return self._clean_sql(sql)

    def _clean_sql(self, sql):
        sql = re.sub(r"```(?:sql)?", "", sql).replace("```", "").strip().rstrip(";")
        sql_upper = sql.upper()
        if not sql_upper.startswith("SELECT"):
            raise AppError("模型生成的语句不是查询语句,已拒绝执行", code=400)
        if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|SHOW|DESCRIBE)\b", sql_upper):
            raise AppError("模型生成的语句包含危险操作,已拒绝执行", code=400)
        if ";" in sql:
            raise AppError("检测到多语句内容,已拒绝执行", code=400)
        if re.search(r"/\*|--\s", sql):
            raise AppError("检测到 SQL 注释,已拒绝执行", code=400)
        if re.search(r"0x[0-9a-fA-F]+", sql):
            raise AppError("检测到十六进制字面量,已拒绝执行", code=400)
        if re.search(r"\b(CONCAT|CHAR|LOAD_FILE|BENCHMARK|SLEEP|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b", sql_upper):
            raise AppError("检测到危险函数,已拒绝执行", code=400)
        if not re.search(r"\bLIMIT\s+\d+", sql_upper):
            sql += " LIMIT 50"
        return sql

    # ============ RAG 流式问答 ============

    def chat_stream(self, query, top_k=3):
        import time
        started = time.time()
        rewrite_failed = False
        try:
            rewritten = self.ollama.chat(
                "把下面问题改写为更清晰的检索问句,只输出改写结果一句:" + query,
                system="你是查询改写助手,只输出一句改写结果,不要解释。",
                temperature=0.1,
            ).strip()
            if not rewritten:
                rewritten = query
                rewrite_failed = True
        except Exception:
            rewritten = query
            rewrite_failed = True
        yield {"type": "meta", "query_rewritten": rewritten, "rewrite_failed": rewrite_failed}
        results = self.search(query, top_k=max(top_k * 2, 6))
        selected = results[:top_k]
        for i, r in enumerate(selected):
            yield {"type": "source", "index": i, "file_name": r.get("file_name", ""),
                   "score": round(r.get("score", 0), 3),
                   "content": r.get("document", "")[:200]}
        context_parts = []
        for r in selected:
            context_parts.append("[来源:" + str(r.get("file_name", "?")) + "] " + str(r.get("document", "")))
        context = chr(10).join(context_parts)
        user_prompt = "参考资料:" + context[:3000] + chr(10) + chr(10) + "问题:" + query
        full_answer = ""
        try:
            for chunk in self.ollama.chat_stream(user_prompt, system=(
                "你是基于参考资料的中文助手。回答用中文,直接给出答案,如果参考资料不足就说未找到相关资料。"), temperature=0.3):
                if chunk:
                    full_answer += chunk
                    yield {"type": "delta", "content": chunk}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return
        yield {"type": "done", "elapsed": round(time.time() - started, 2),
               "answer": full_answer, "sources": selected}

    # ============ 简历与岗位匹配 ============

    def _build_jd_text(self, job):
        parts = [
            "岗位名称:" + str(job.get("job_name", "")),
            "公司:" + str(job.get("company", "")),
            "城市:" + str(job.get("city", "")),
            "薪资:" + str(job.get("salary_min", 0)) + "K-" + str(job.get("salary_max", 0)) + "K",
            "学历:" + str(job.get("education", "")),
            "经验:" + str(job.get("experience", "")),
            "技能:" + str(job.get("skill_require", "")),
            "描述:" + str(job.get("job_desc", "")),
        ]
        return chr(10).join(p for p in parts if p.split(":", 1)[-1].strip())

    def _extract_skill_keywords(self, text):
        found = []
        for kw in SKILL_KEYWORDS:
            if kw.lower() in (text or "").lower():
                found.append(kw)
        return found

    def _llm_match_score(self, resume_text, jd_text):
        system_prompt = (
            "你是资深的 HR 岗位匹配专家。请根据简历文本与岗位 JD,从以下维度打分(每项 0-100):"
            "技能匹配、经验匹配、学历匹配、项目契合。"
            "必须严格输出如下 JSON 格式(不要任何多余内容):"
            '{"total_score": 85, "dimensions": {"技能匹配": 90, "经验匹配": 80, "学历匹配": 100, "项目契合": 70}, "description": "整体匹配说明(100字内)"}'
        )
        user_prompt = "【简历】" + chr(10) + (resume_text or "")[:3000] + chr(10) + chr(10) + "【JD】" + chr(10) + (jd_text or "")[:3000]
        try:
            raw = self.ollama.chat(user_prompt, system=system_prompt, temperature=0.2)
            data = extract_json_from_text(raw)
            dimensions = {str(k): int(v) for k, v in (data.get("dimensions") or {}).items()}
            if not dimensions:
                raise AppError("评分维度为空")
            total = int(data.get("total_score") or round(sum(dimensions.values()) / len(dimensions)))
            return {"total_score": total, "dimensions": dimensions, "description": str(data.get("description", ""))}
        except (AppError, ValueError):
            logger.warning("LLM 匹配评分解析失败,使用规则评分兜底")
            return self._rule_match_score(resume_text, jd_text)

    def _rule_match_score(self, resume_text, jd_text):
        resume_lower = (resume_text or "").lower()
        skill_keywords = self._extract_skill_keywords(jd_text)
        if not skill_keywords:
            skill_score = 80.0
        else:
            hit = sum(1 for w in skill_keywords if w.lower() in resume_lower)
            skill_score = round(hit / len(skill_keywords) * 100)
        exp_keywords = ["应届", "1-3年", "3-5年", "5年以上"]
        exp_hit = sum(1 for w in exp_keywords if w in jd_text and w in resume_text)
        exp_score = 60 + exp_hit * 20 if exp_hit else 70
        edu_keywords = ["本科", "硕士", "博士", "大专"]
        edu_score = 80 if any(w in jd_text and w in resume_text for w in edu_keywords) else 70
        proj_score = min(skill_score + 5, 100)
        dimensions = {
            "技能匹配": skill_score, "经验匹配": exp_score,
            "学历匹配": edu_score, "项目契合": proj_score,
        }
        return {
            "total_score": int(round(sum(dimensions.values()) / len(dimensions))),
            "dimensions": dimensions,
            "description": "规则评分兜底结果(LLM 评分失败时使用)",
        }

    def generate_match_chart(self, dimensions):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            labels = list(dimensions.keys())
            # 强制 clip 到 0-100,避免 matplotlib 文本标注超界
            scores = [max(0, min(100, v)) for v in dimensions.values()]
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#378ADD", "#639922", "#BA7517", "#7F77DD", "#D85A30"][: len(labels)]
            bars = ax.bar(labels, scores, color=colors)
            ax.set_ylim(0, 100)
            ax.set_ylabel("得分")
            ax.set_title("简历-岗位匹配度")
            for b, s in zip(bars, scores):
                ax.text(b.get_x() + b.get_width() / 2, s + 2, str(int(s)),
                        ha="center", va="bottom", fontsize=11)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.warning("图表生成失败:%s", exc)
            return ""

    def match_resume(self, resume_text, job_id):
        job = self.mysql.query_one("SELECT * FROM job_info WHERE id = %s", (job_id,))
        if not job:
            raise AppError("岗位 ID=" + str(job_id) + " 不存在", code=404)
        jd_text = self._build_jd_text(job)
        result = self._llm_match_score(resume_text, jd_text)
        result["job"] = {k: v for k, v in job.items() if k != "create_time"}
        result["chart_base64"] = self.generate_match_chart(result["dimensions"])
        result["suggestions"] = self._generate_suggestions(resume_text, jd_text, result)
        return result

    def match_resume_with_jd(self, resume_text, jd_text, target_job=""):
        """用户上传 JD 文本 + 简历 → 精准匹配评分

        适用场景:用户在 Boss/拉勾看到具体岗位,粘 JD 原文进来,想知道简历能不能过关
        """
        if len((resume_text or "").strip()) < 10:
            raise AppError("简历内容过短,请粘贴完整文本或上传 PDF/DOCX", code=400)
        if len((jd_text or "").strip()) < 20:
            raise AppError("JD 文本过短,请粘贴完整的岗位描述（含职责/技能要求）", code=400)
        result = self._llm_match_score(resume_text, jd_text)
        # 构造展示用 job 字段
        result["job"] = {
            "job_name": target_job or "（用户上传的 JD）",
            "company": "——",
            "city": "——",
            "salary_min": 0,
            "salary_max": 0,
        }
        result["chart_base64"] = self.generate_match_chart(result["dimensions"])
        result["suggestions"] = self._generate_suggestions(resume_text, jd_text, result)
        return result

    def _generate_suggestions(self, resume_text, jd_text, score_result):
        """根据评分结果给具体调整建议(总分 < 85 时给出 actionable 建议)"""
        total = score_result.get("total_score", 0)
        if total >= 85:
            return ["综合匹配度较高,建议直接投递并准备面试"]
        # 找出最弱的维度
        dims = score_result.get("dimensions") or {}
        weak = sorted(dims.items(), key=lambda x: x[1])[:2]
        weak_str = "、".join(f"{k}({v}分)" for k, v in weak)
        suggestions = [
            f"综合匹配度 {total} 分,未达 85 分及格线",
            f"较弱维度:{weak_str},建议针对性补充",
        ]
        # 用 LLM 给出更具体的建议
        try:
            system_prompt = (
                "你是求职顾问。基于简历文本和岗位 JD,以及初步评分结果,给出 3 条最具体的简历调整建议。"
                "每条建议不超过 30 字,直接说怎么改(如『补充 Python 项目经验』『增加 Kafka 实战案例』)。"
                "只输出 3 条建议的纯文本,每条一行,不要其他内容。"
            )
            user_prompt = (
                f"【简历】\n{resume_text[:2000]}\n\n【JD】\n{jd_text[:2000]}\n\n"
                f"【初步评分】总分 {total} 分,各维度:{dims}\n"
                f"【最弱维度】{weak_str}"
            )
            raw = self.ollama.chat(user_prompt, system=system_prompt, temperature=0.3)
            lines = [line.strip().lstrip("•-0123456789. ").strip() for line in raw.split("\n") if line.strip()]
            llm_suggestions = [l for l in lines if 5 <= len(l) <= 60][:3]
            if llm_suggestions:
                suggestions.extend(llm_suggestions)
        except Exception as exc:
            logger.warning("生成调整建议失败,使用规则兜底: %s", exc)
        return suggestions

    def match_resume_to_target(self, resume_text, target_job, city="", salary_min=0, top_n=5):
        if not (target_job or "").strip():
            raise AppError("请输入目标岗位名称", code=400)
        if len((resume_text or "").strip()) < 10:
            raise AppError("简历内容过短,请粘贴完整文本或上传 PDF/DOCX", code=400)
        search = self.search_jobs(
            keywords=target_job.strip(),
            city=(city or "").strip(),
            salary_min=int(salary_min or 0),
            limit=max(3, top_n),
        )
        candidates = search["items"]
        sources_used = search["sources_used"]
        sources_failed = search.get("sources_failed", {})
        if not candidates:
            return {
                "target_job": target_job, "city": city,
                "salary_min": int(salary_min or 0),
                "candidates_count": 0, "matched": None, "alternatives": [],
                "sources_used": sources_used, "sources_failed": sources_failed,
                "message": "本地与外部数据源均无该岗位匹配;可调整关键词/薪资,或粘贴 JD 入库。",
            }
        scored = []
        for job in candidates:
            jd_text = self._build_jd_text(job)
            try:
                score = self._llm_match_score(resume_text, jd_text)
            except Exception as exc:
                logger.warning("评分失败 job=%s: %s", job.get("id"), exc)
                continue
            scored.append({
                "job": {k: v for k, v in job.items() if k not in ("create_time",)},
                "total_score": score["total_score"],
                "dimensions": score["dimensions"],
                "description": score["description"],
            })
        if not scored:
            raise AppError("所有候选 JD 均评分失败,请重试", code=500)
        scored.sort(key=lambda x: x["total_score"], reverse=True)
        best = scored[0]
        alts = scored[1:4]
        chart = self.generate_match_chart(best["dimensions"])
        # 给最佳匹配生成调整建议
        best_job_jd = self._build_jd_text(best["job"])
        best["suggestions"] = self._generate_suggestions(resume_text, best_job_jd, best)
        # 及格分阈值:85
        pass_line = 85
        if best["total_score"] >= pass_line:
            overall = (
                "简历与「" + target_job + "」匹配度最高的是【"
                + best["job"]["company"] + "】的【"
                + best["job"]["job_name"] + "】(城市 "
                + str(best["job"].get("city", "?")) + ",总分 "
                + str(best["total_score"]) + " 分) ✅ 达到 85 分及格线,建议直接投递并准备面试。"
            )
        else:
            overall = (
                "简历与「" + target_job + "」匹配度最高的是【"
                + best["job"]["company"] + "】的【"
                + best["job"]["job_name"] + "】(总分 "
                + str(best["total_score"]) + " 分) ⚠️ 未达 85 分及格线,建议参考下方调整建议优化简历后再投递。"
            )
        if sources_used:
            overall += " 数据来源:" + ", ".join(sources_used) + "。"
        return {
            "target_job": target_job, "city": city,
            "salary_min": int(salary_min or 0),
            "candidates_count": len(candidates), "scored_count": len(scored),
            "matched": {**best, "chart_base64": chart},
            "alternatives": alts,
            "sources_used": sources_used, "sources_failed": sources_failed,
            "overall_suggestion": overall,
            "pass_line": pass_line,
        }

    # ============ 多源实时检索 ============

    def search_jobs(self, keywords, city="", salary_min=0, sources=None, limit=30):
        import time
        if not (keywords or "").strip():
            raise AppError("关键词不能为空", code=400)
        # 默认行为:local(样例库) + tencent(腾讯公开招聘),adzuna 可选
        # 每个数据源会自己判断 is_available(),配了 token/key 才生效
        if sources is None:
            sources = ["local", "tencent", "adzuna"]
        started = time.time()
        local_count = 0
        external_added = 0
        external_skipped = 0
        sources_used = []
        sources_failed = {}

        local_results = []
        if "local" in sources:
            local_results = self._search_local_jobs(keywords, city, salary_min, limit)
            local_count = len(local_results)
            sources_used.append("local")

        # 短路条件:仅当"local 命中已满"或"只有 local 一个源"时直接返回
        # (修复:只选 tencent/adzuna 而不含 local 时,不能短路返回空)
        if local_count >= limit or (len(sources) == 1 and "local" in sources):
            # 即使命中已满,也触发一次腾讯源节流刷新(增量更新库,不阻塞返回)
            if "tencent" in sources:
                self._maybe_refresh_tencent(keywords.strip(), city or "")
            return {
                "items": local_results[:limit], "total": min(local_count, limit),
                "sources_used": sources_used, "local_count": local_count,
                "external_count": 0, "external_skipped": 0,
                "sources_failed": {}, "elapsed_seconds": round(time.time() - started, 2),
                "params": {"keywords": keywords, "city": city, "salary_min": salary_min, "limit": limit},
            }

        for src_name in sources:
            if src_name == "local":
                continue
            from app.dao.external_api import SOURCE_REGISTRY, get_source
            if src_name not in SOURCE_REGISTRY:
                sources_failed[src_name] = "未知数据源"
                continue
            src = get_source(src_name)
            if not src.is_available():
                sources_failed[src_name] = "未配置或鉴权失败"
                continue
            try:
                raw_jobs = src.fetch_jobs(
                    limit=max(10, limit - local_count),
                    what=keywords.strip(),
                    where=(city or "").strip(),
                    salary_min=int(salary_min or 0),
                )
            except Exception as exc:
                sources_failed[src_name] = str(exc)
                continue
            sources_used.append(src_name)
            added, skipped = self._batch_upsert_external(raw_jobs, src, salary_min)
            external_added += added
            external_skipped += skipped

        # 最终合并查询:严格按 sources 过滤 data_source(修复:只选 external 时不混入 local)
        active_sources = [s for s in sources if s in ("local",) or s in SOURCE_REGISTRY]
        merged = self._search_local_jobs(keywords, city, salary_min, limit, data_sources=active_sources)
        seen = set()
        unique = []
        for row in merged:
            key2 = (row.get("data_source", "local"), row.get("external_id", ""))
            if key2 in seen:
                continue
            seen.add(key2)
            unique.append(row)
            if len(unique) >= limit:
                break
        return {
            "items": unique, "total": len(unique),
            "sources_used": sources_used, "local_count": local_count,
            "external_count": external_added, "external_skipped": external_skipped,
            "sources_failed": sources_failed,
            "elapsed_seconds": round(time.time() - started, 2),
            "params": {"keywords": keywords, "city": city, "salary_min": salary_min, "limit": limit},
        }

    def _maybe_refresh_tencent(self, keywords: str = "", city: str = ""):
        """腾讯源节流增量刷新：距上次成功刷新超过 REFRESH_INTERVAL 秒则拉取一次关键词相关岗位入库

        目的：让本地腾讯岗位库保持新鲜（不随每次搜索都打 API，也不长期不更新）
        """
        import time as _t
        interval = 1800  # 30 分钟
        now = _t.time()
        if getattr(self, "_last_tencent_refresh", 0) and (now - self._last_tencent_refresh) < interval:
            return
        try:
            from app.dao.external_api import get_source
            src = get_source("tencent")
            if not src.is_available():
                return
            raw_jobs = src.fetch_jobs(limit=30, what=keywords or "AI", where=city)
            if raw_jobs:
                added, skipped = self._batch_upsert_external(raw_jobs, src, 0)
                self._last_tencent_refresh = now
                logger.info("腾讯源节流刷新: 拉取 %d 条, 新增 %d, 跳过 %d", len(raw_jobs), added, skipped)
        except Exception as exc:
            logger.warning("腾讯源节流刷新失败(不影响检索): %s", exc)

    def _search_local_jobs(self, keywords, city, salary_min, limit, data_sources=None):
        wheres, params = [], []
        if data_sources:
            placeholders = ",".join(["%s"] * len(data_sources))
            wheres.append(f"data_source IN ({placeholders})")
            params.extend(list(data_sources))
        if (keywords or "").strip():
            wheres.append("(job_name LIKE %s OR skill_require LIKE %s OR job_desc LIKE %s)")
            like = "%" + keywords.strip() + "%"
            params.extend([like, like, like])
        if (city or "").strip():
            wheres.append("city LIKE %s")
            params.append("%" + city.strip() + "%")
        if salary_min and salary_min > 0:
            wheres.append("salary_max > %s")
            params.append(int(salary_min))
        where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""
        sql = (
            "SELECT id, job_name, company, city, salary_min, salary_max, "
            "education, experience, skill_require, data_source, external_id, source_url, "
            "create_time FROM job_info " + where_clause + " ORDER BY create_time DESC LIMIT %s"
        )
        params.append(int(limit))
        exec_sql = sql.replace("%", SQL_PCT_PLACEHOLDER)
        return self.mysql.query(exec_sql, tuple(params))

    def _batch_upsert_external(self, raw_jobs, src, salary_min=0):
        added = 0
        skipped = 0
        for raw in raw_jobs:
            try:
                row = src.to_job_info(raw)
                if not row.get("job_name") or not row.get("company"):
                    skipped += 1
                    continue
                if salary_min and salary_min > 0:
                    s_max = row.get("salary_max") or 0
                    if s_max and s_max <= salary_min:
                        skipped += 1
                        continue
                exists = self.mysql.query_one(
                    "SELECT id FROM job_info WHERE data_source = %s AND external_id = %s",
                    (src.name, row.get("external_id", "")),
                )
                if exists:
                    skipped += 1
                    continue
                self.mysql.insert_one(
                    "INSERT INTO job_info (job_name, company, city, salary_min, salary_max, "
                    "skill_require, education, experience, job_desc, "
                    "data_source, external_id, source_url) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        row["job_name"], row["company"], row["city"],
                        row["salary_min"], row["salary_max"],
                        row["skill_require"], row["education"], row["experience"],
                        row["job_desc"], src.name, row.get("external_id", ""),
                        row.get("source_url", ""),
                    ),
                )
                added += 1
            except Exception as exc:
                logger.warning("[%s] upsert 失败:%s", src.name, exc)
                skipped += 1
        return added, skipped

    # ============ 外部导入 + 粘贴入库 ============

    def list_external_sources(self) -> dict:
        """列出全部可用的外部数据源(前端选择用)"""
        from app.dao.external_api import list_sources
        return {"sources": list_sources()}

