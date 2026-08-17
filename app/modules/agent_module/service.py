"""简历评分优化 Agent - 业务逻辑层（ReAct 多轮工具调用）

职责：
1. ReAct Agent 实例构建：绑定全部自定义工具，仅使用本地 Ollama 模型
2. Agent 执行入口：接收简历文件 + 目标岗位，驱动多轮「思考→行动→观察」调用链
3. 结果格式化：原始输出整理为结构化数据（总分/分项维度得分/弱点清单/优化建议/修改示例）
4. 安全与异常：调用链超时控制、最大工具轮数限制、Ollama 失败降级、
   解析失败自动纠错/兜底编排，保证接口始终返回结构化结果
5. 会话记忆：复用 agent_memory 缓存中间结果，避免重复调用大模型

分层约束：本层只做业务编排，工具实现见 tools.py，LLM 调用走通用 utils/ollama_client.py。
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from app.modules.agent_module import parser as agent_parser
from app.modules.agent_module import tools as agent_tools
from app.modules.agent_module.config import (
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_ROUND_SECONDS,
    AGENT_OBSERVATION_TEMPLATE,
    AGENT_SYSTEM_PROMPT,
    AGENT_TEMPERATURE,
    AGENT_TOOLS_DESC,
    DIMENSION_MIN,
    LLM_FALLBACK_THRESHOLD,
    PASS_LINE,
    SCORE_WEIGHTS,
    TOTAL_FLOOR,
)
from app.modules.agent_module.memory import AgentMemory
from app.utils.common_tools import AppError, get_logger
from app.utils.ollama_client import OllamaClient

logger = get_logger("agent_service")

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class AgentLoopError(AppError):
    """Agent 调用链异常（超时/超轮数/解析失败），触发降级"""


class ResumeAgentService:
    """简历评分优化 Agent 服务（单例）"""

    _instance: Optional["ResumeAgentService"] = None

    def __new__(cls) -> "ResumeAgentService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.memory = AgentMemory()

    # ==================== 文件校验 ====================

    def _validate_file(self, filename: str) -> None:
        """文件校验：仅允许 PDF / DOCX"""
        ext = Path(filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise AppError("仅支持 PDF / DOCX 格式简历文件", code=400)

    # ==================== ReAct Agent 构建 ====================

    def _build_agent(self, session) -> dict:
        """构建 ReAct Agent 运行时：绑定全部工具并注入会话级缓存

        缓存策略：同一会话内 resume_struct / job_requirements 已获取过则直接复用，
        避免重复调用大模型。

        Args:
            session: AgentSession 会话对象
        Returns:
            {"tools": {工具名: cached_run}, "max_iterations": int}
        """
        cached_tools: dict = {}

        def _cached_run(tool, args: dict) -> str:
            """带会话缓存的工具执行器"""
            # 缓存命中直接返回
            if tool.name == "tool_resume_parser" and session.resume_struct is not None:
                return json.dumps(session.resume_struct, ensure_ascii=False)
            if tool.name == "tool_get_job_requirements" and session.job_requirements is not None:
                return json.dumps(session.job_requirements, ensure_ascii=False)
            result_text = tool.run(args)
            # 写缓存（解析为 dict 后存储）
            try:
                parsed = json.loads(result_text)
                if tool.name == "tool_resume_parser" and isinstance(parsed, dict):
                    session.resume_struct = parsed
                if tool.name == "tool_get_job_requirements" and isinstance(parsed, dict):
                    session.job_requirements = parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return result_text

        for name, tool in agent_tools.AGENT_TOOLS.items():
            cached_tools[name] = lambda args, _tool=tool: _cached_run(_tool, args)

        return {"tools": cached_tools, "max_iterations": AGENT_MAX_ITERATIONS}

    # ==================== Agent 执行入口 ====================

    def evaluate(
        self,
        file_bytes: bytes,
        filename: str,
        target_job: str,
        session_id: Optional[str] = None,
        on_event: Optional[Any] = None,
        jd_text: str = "",
    ) -> dict:
        """Agent 完整评估入口：简历解析 -> ReAct 多轮工具调用 -> 结构化结果

        Args:
            file_bytes: 简历文件二进制
            filename: 简历文件名
            target_job: 目标岗位名称
            session_id: 会话 ID（复用中间结果，缺省新建）
            on_event: 过程事件回调（思考/行动/观察），用于流式展示
        Returns:
            结构化评估结果 dict
        """
        self._validate_file(filename)
        session = self.memory.get_or_create(session_id)
        session.task = {"filename": filename, "target_job": target_job}

        # 保存简历文件并提取纯文本（复用 doc_parser，仅一次）
        resume_text = self._extract_resume_text(file_bytes, filename)
        session.task["resume_text"] = resume_text
        session.touch()

        # 简历内容过少校验:可能不是有效简历(乱码/空文档/非简历内容)
        if len(resume_text.strip()) < 100:
            raise AppError("简历内容过少,可能不是有效简历,请上传包含完整个人信息、技能与项目经历的简历", code=400)

        # 统一预取岗位要求:JD 模式直接用,关键词模式走 RAG 检索
        # 每次按当前输入重新获取(不复用旧会话缓存 - target_job/jd_text 可能变化)
        session.job_requirements = agent_tools.tool_get_job_requirements(target_job, jd_text)

        # 若已有最终结果且会话未变，直接复用（避免重复调用大模型）
        if session.final_result and session.final_result.get("task") == (filename, target_job):
            return session.final_result["data"]

        try:
            result = self._run_react(session, resume_text, target_job, on_event)
        except AgentLoopError as exc:
            logger.warning("ReAct 调用链降级：%s", exc)
            if on_event:
                on_event({"type": "fallback", "message": "已使用基础评估模式完成分析"})
            result = self._fallback_evaluate(session, resume_text, target_job)
        except AppError as exc:
            logger.error("Agent 评估失败：%s", exc)
            raise

        session.final_result = {"task": (filename, target_job), "data": result}
        session.touch()
        return result

    def _extract_resume_text(self, file_bytes: bytes, filename: str) -> str:
        """复用 doc_parser 提取简历纯文本"""
        from app.utils.doc_parser import parse_document

        ext = Path(filename).suffix.lower()
        save_path = _temp_save(file_bytes, ext)
        try:
            parsed = parse_document(save_path)
            return parsed["text"]
        finally:
            try:
                Path(save_path).unlink(missing_ok=True)
            except OSError:
                pass

    # ==================== ReAct 多轮循环 ====================

    def _run_react(self, session, resume_text: str, target_job: str, on_event=None,
                   temperature: Optional[float] = None, max_iterations: Optional[int] = None) -> dict:
        """驱动 ReAct 多轮工具调用链路

        轮数限制：AGENT_MAX_ITERATIONS（默认 6，可被参数覆盖）
        整体超时：AGENT_MAX_ROUND_SECONDS（默认 300s）
        防重复调用：同一工具最多调用 2 次
        """
        agent = self._build_agent(session)
        max_iterations = max_iterations or agent["max_iterations"]
        temperature = temperature if temperature is not None else AGENT_TEMPERATURE
        tools_map = agent["tools"]
        start_time = time.time()

        system_prompt = AGENT_SYSTEM_PROMPT.format(tools_desc=AGENT_TOOLS_DESC)
        task_prompt = (
            f"任务：对以下简历进行岗位定向评估。\n\n"
            f"目标岗位：{target_job or '（未指定，进行通用评估）'}\n\n"
            f"简历文本（前3000字）：\n{resume_text[:3000]}"
        )
        history = [{"role": "user", "content": task_prompt}]
        tool_call_counts: dict = {}

        for iteration in range(1, max_iterations + 1):
            # 整体调用链超时控制
            if time.time() - start_time > AGENT_MAX_ROUND_SECONDS:
                raise AgentLoopError(f"Agent 调用链超过 {AGENT_MAX_ROUND_SECONDS}s 超时上限")

            # 单轮 LLM 推理（超时/重试由 ollama_client 统一处理）
            raw = self.ollama.chat(
                "", system=system_prompt, messages=history, temperature=temperature
            )
            history.append({"role": "assistant", "content": raw})

            # 解析 ReAct 输出（失败抛出 AppError，进入降级）
            try:
                parsed = agent_parser.parse_react_output(raw)
            except AppError as exc:
                raise AgentLoopError(f"ReAct 输出解析失败：{exc.message}") from exc

            # 评估完成
            if "final_answer" in parsed:
                if on_event:
                    on_event({"type": "final", "iteration": iteration, "thought": parsed.get("thought", "")})
                return self._format_result(session, parsed.get("final_answer", {}), target_job, resume_text)

            # 工具调用
            action = parsed["action"]
            action_input = parsed.get("action_input") or {}
            # 防重复调用保护：同一工具最多 2 次
            tool_call_counts[action] = tool_call_counts.get(action, 0) + 1
            if tool_call_counts[action] > 2:
                raise AgentLoopError(f"工具 {action} 重复调用超过 2 次，疑似循环")

            if action not in tools_map:
                raise AgentLoopError(f"未知工具：{action}")

            observation = tools_map[action](action_input)
            if on_event:
                on_event(
                    {
                        "type": "step",
                        "iteration": iteration,
                        "thought": parsed.get("thought", ""),
                        "action": action,
                        "observation": observation[:500],
                    }
                )
            history.append(
                {
                    "role": "user",
                    "content": AGENT_OBSERVATION_TEMPLATE.format(observation=observation[:1500]),
                }
            )

        raise AgentLoopError(f"超过最大工具调用轮数 {max_iterations}，触发保护")

    # ==================== 结果格式化 ====================

    def _format_result(self, session, final_answer: Any, target_job: str, resume_text: str) -> dict:
        """把 Agent 原始输出整理为结构化数据(锚定评分:优秀=95 标杆,严格化)

        关键:总分 AND 维度 都用工具计算(不用 LLM final_answer 的分数),
        避免 LLM 乱打高分导致"总分 95 + 维度 80"的割裂现象。
        """
        if not isinstance(final_answer, dict):
            final_answer = {"total_score": 0, "dimensions": {}, "weaknesses": [], "suggestions": []}

        # 维度统一用工具计算(规则稳定可复现,不依赖 LLM 主观评分)
        jd_text = (session.job_requirements or {}).get("jd", "")
        info = session.resume_struct or agent_tools.tool_resume_parser(resume_text=resume_text)
        if session.resume_struct is None:
            session.resume_struct = info
        rule_score = agent_tools.tool_calc_match_score(
            resume_info=info, job_requirements=jd_text, resume_text=resume_text
        )
        session.score = rule_score
        dimensions = {k: max(DIMENSION_MIN, v) for k, v in (rule_score.get("dimensions") or {}).items()}

        # 总分:用更严格的 benchmark(必须真正优秀才能 90+)
        total = self._benchmark_total(session, resume_text)

        weaknesses = final_answer.get("weaknesses") or _auto_weaknesses(dimensions)
        suggestions = final_answer.get("suggestions") or []
        # **建议数扣分**:让分数与"还需改"挂钩
        suggestion_penalty = min(len(suggestions) * 3, 15)
        total = max(total - suggestion_penalty, 30)

        jd_result = session.job_requirements or {}
        score = {
            "total_score": total,
            "dimensions": dimensions,
            "matched_skills": rule_score.get("matched_skills", []),
            "missing_skills": rule_score.get("missing_skills", []),
        }
        session.score = score

        # 若模型未生成建议，用工具补齐（缓存）
        if not suggestions:
            suggestion = self._generate_suggestion_cached(session, resume_text)
            suggestions = suggestion.get("suggestions", [])
            summary = suggestion.get("summary", "")
        else:
            summary = final_answer.get("summary", "")

        return {
            "session_id": session.session_id,
            "mode": "岗位定向评估" if jd_result.get("found") else "通用评估",
            "job_info": jd_result.get("job") if isinstance(jd_result, dict) else None,
            "jd_message": (jd_result or {}).get("message", ""),
            "candidates": (jd_result or {}).get("candidates") or [],
            "used_job_name": (jd_result or {}).get("job_name") if (jd_result or {}).get("found") else "",
            "total_score": total,
            "dimensions": dimensions,
            "weaknesses": [str(w) for w in weaknesses],
            "suggestions": suggestions,
            "summary": summary,
            "matched_skills": score["matched_skills"],
            "missing_skills": score["missing_skills"],
            "pass_line": PASS_LINE,
            "pass": total >= PASS_LINE,
            "score_source": "benchmark",
            "task": {"filename": (session.task or {}).get("filename", ""), "target_job": target_job},
        }

    def _generate_suggestion_cached(self, session, resume_text: str) -> dict:
        """生成优化建议（会话内缓存，避免重复调用大模型）"""
        if session.suggestion is not None:
            return session.suggestion
        detail = {
            "dimensions": (session.score or {}).get("dimensions", {}),
            "missing_skills": (session.score or {}).get("missing_skills", []),
            "weaknesses": (session.final_result or {}).get("data", {}).get("weaknesses", []),
        }
        job_req = (session.job_requirements or {}).get("jd", "")
        suggestion = agent_tools.tool_generate_suggestion(
            resume_text=resume_text, score_detail=detail, job_requirements=job_req
        )
        session.suggestion = suggestion
        return suggestion

    def _benchmark_total(self, session, resume_text: str) -> int:
        """锚定总分：以「优秀简历=90 分」为标杆计算（所有评估路径统一）"""
        try:
            info = session.resume_struct
            if info is None:
                info = agent_tools.tool_resume_parser(resume_text=resume_text)
                session.resume_struct = info
            return agent_tools.tool_benchmark_score(
                resume_info=info,
                job_requirements=(session.job_requirements or {}).get("jd", ""),
                resume_text=resume_text,
            )
        except Exception as exc:
            logger.warning("锚定评分失败,回退规则评分: %s", exc)
            rule_score = agent_tools.tool_calc_match_score(
                resume_info=session.resume_struct or agent_tools.tool_resume_parser(resume_text=resume_text),
                job_requirements=(session.job_requirements or {}).get("jd", ""),
                resume_text=resume_text,
            )
            return int(rule_score.get("total_score", 60))

    # ==================== 兜底编排（降级路径） ====================

    def _fallback_evaluate(self, session, resume_text: str, target_job: str) -> dict:
        """ReAct 异常时兜底：直接编排调用工具链，保证返回结构化结果"""
        logger.info("Agent 兜底编排开始 target_job=%s", target_job)

        # 1) 获取岗位要求（RAG 联动）
        if session.job_requirements is None:
            session.job_requirements = agent_tools.tool_get_job_requirements(target_job)

        # 2) 解析简历
        if session.resume_struct is None:
            session.resume_struct = agent_tools.tool_resume_parser(resume_text=resume_text)

        # 3) 匹配打分（规则，稳定）：维度展示用规则分，总分用锚定评分(优秀=90 标杆)
        score = agent_tools.tool_calc_match_score(
            resume_info=session.resume_struct,
            job_requirements=(session.job_requirements or {}).get("jd", ""),
            resume_text=resume_text,
        )
        session.score = score
        dimensions = {k: max(DIMENSION_MIN, v) for k, v in (score.get("dimensions") or {}).items()}
        total = self._benchmark_total(session, resume_text)
        weaknesses = [f"缺少技能：{s}" for s in (score.get("missing_skills") or [])[:4]]
        weaknesses += [f"{k}得分偏低（{v}分）" for k, v in dimensions.items() if v < PASS_LINE]
        if not weaknesses:
            weaknesses = ["整体匹配度尚可，建议进一步突出项目成果量化指标"]

        # 4) 生成优化建议
        suggestion = self._generate_suggestion_cached(session, resume_text)
        # 5) 一句话总结（LLM 生成,失败用规则文本）
        summary = self._summary_from_llm(resume_text, session.job_requirements, total=total)
        # 6) **建议数扣分**:让分数与"还需改"挂钩——6 条建议就该 70 出头,不是 85+
        suggestions = suggestion.get("suggestions", [])
        suggestion_penalty = min(len(suggestions) * 3, 15)
        total = max(total - suggestion_penalty, 30)

        return {
            "session_id": session.session_id,
            "mode": "岗位定向评估" if (session.job_requirements or {}).get("found") else "通用评估",
            "job_info": (session.job_requirements or {}).get("job"),
            "jd_message": (session.job_requirements or {}).get("message", ""),
            "candidates": (session.job_requirements or {}).get("candidates") or [],
            "used_job_name": (session.job_requirements or {}).get("job_name") if (session.job_requirements or {}).get("found") else "",
            "total_score": total,
            "dimensions": dimensions,
            "weaknesses": weaknesses,
            "suggestions": suggestion.get("suggestions", []),
            "summary": summary,
            "matched_skills": score.get("matched_skills", []),
            "missing_skills": score.get("missing_skills", []),
            "pass_line": PASS_LINE,
            "pass": total >= PASS_LINE,
            "score_source": "benchmark",
            "task": {"filename": (session.task or {}).get("filename", ""), "target_job": target_job},
            "fallback": True,
        }

    def _summary_from_llm(self, resume_text: str, job_req: Optional[dict], total: int) -> str:
        """生成整体评价（LLM，失败回退规则文本）"""
        try:
            raw = self.ollama.chat(
                f"【简历文本】\n{resume_text[:2000]}\n\n【岗位JD】\n{(job_req or {}).get('jd', '')[:1500]}",
                system=(
                    "你是资深 HR。请用 80 字以内总结简历与岗位的整体匹配情况，客观指出主要亮点与短板。"
                    "只输出总结文本，不要任何格式标记。"
                ),
                temperature=0.3,
            )
            return raw.strip()[:200] or f"综合匹配度 {total} 分。"
        except AppError:
            return f"综合匹配度 {total} 分，建议重点提升低分维度。"

    # ==================== 单步能力（接口直调） ====================

    def parse_resume(self, file_bytes: bytes, filename: str) -> dict:
        """简历解析（复用 doc_parser + LLM 结构化）"""
        self._validate_file(filename)
        resume_text = self._extract_resume_text(file_bytes, filename)
        return agent_tools.tool_resume_parser(resume_text=resume_text)

    def score(self, resume_text: str, job_desc: Optional[str] = None) -> dict:
        """简历-JD 多维度匹配打分（规则，稳定可复现）"""
        info = agent_tools.tool_resume_parser(resume_text=resume_text)
        score = agent_tools.tool_calc_match_score(
            resume_info=info, job_requirements=job_desc or "", resume_text=resume_text
        )
        return {
            "total_score": score["total_score"],
            "dimensions": score["dimensions"],
            "matched_skills": score.get("matched_skills", []),
            "missing_skills": score.get("missing_skills", []),
            "mode": "岗位定向评分" if job_desc else "通用评分",
        }

    def get_suggestion(self, resume_text: str, target_job: Optional[str] = None, score_detail: dict = None) -> dict:
        """单独获取优化建议（可选接口）"""
        job_req = ""
        if target_job:
            job_req = agent_tools.tool_get_job_requirements(target_job).get("jd", "")
        detail = score_detail or {}
        return agent_tools.tool_generate_suggestion(
            resume_text=resume_text, score_detail=detail, job_requirements=job_req
        )

    # ==================== 会话管理 ====================

    def clear_session(self, session_id: str) -> dict:
        """清除指定会话"""
        existed = self.memory.clear_session(session_id)
        return {"cleared": existed, "session_id": session_id}

    def clear_all_sessions(self) -> dict:
        """清空全部会话"""
        count = self.memory.clear_all()
        return {"cleared": count}

    # ==================== 流式事件（思考过程） ====================

    def evaluate_stream(self, file_bytes: bytes, filename: str, target_job: str, session_id: Optional[str] = None, jd_text: str = "",
                        temperature: Optional[float] = None, max_iterations: Optional[int] = None):
        """Agent 评估流式事件生成器（SSE）

        事件类型：
        - {"type": "start", "session_id": ...}
        - {"type": "step",  "iteration", "thought", "action", "observation"}   思考/行动/观察
        - {"type": "fallback", "message": ...}                                  降级提示
        - {"type": "result", "data": {...}}                                     最终结构化结果
        - {"type": "error",  "message": ...}                                    错误
        """
        try:
            self._validate_file(filename)
            session = self.memory.get_or_create(session_id)
            yield {"type": "start", "session_id": session.session_id, "target_job": target_job}

            resume_text = self._extract_resume_text(file_bytes, filename)
            session.task = {"filename": filename, "target_job": target_job, "resume_text": resume_text}
            session.touch()

            # 简历内容过少校验(流式接口同样拦截)
            if len(resume_text.strip()) < 100:
                yield {"type": "error", "message": "简历内容过少,可能不是有效简历,请上传包含完整个人信息、技能与项目经历的简历"}
                return

            # 统一预取岗位要求:JD 模式直接用,关键词模式走 RAG 检索
            # 每次按当前输入重新获取(不复用旧会话缓存 - target_job/jd_text 可能变化)
            session.job_requirements = agent_tools.tool_get_job_requirements(target_job, jd_text)
            # 上传 JD 场景:发一个"step"事件让前端展示来源
            if jd_text and jd_text.strip() and len(jd_text.strip()) >= 20:
                    yield {
                        "type": "step", "iteration": 0,
                        "thought": f"用户上传了 JD 文本({len(jd_text.strip())}字),直接用做精准评估,跳过外部 API 搜索",
                        "action": "tool_get_job_requirements(jd_text=...)",
                        "observation": f"✓ JD 已注入:{session.job_requirements.get('message', '')}",
                    }

            # 复用缓存结果
            if session.final_result and session.final_result.get("task") == (filename, target_job):
                yield {"type": "result", "data": session.final_result["data"]}
                return

            result = None
            buffer: list = []

            def step_cb(event: dict) -> None:
                """收集过程事件到 buffer，保证结果事件最后发出"""
                item = dict(event)
                item["session_id"] = session.session_id
                buffer.append(item)

            try:
                result = self._run_react(session, resume_text, target_job, on_event=step_cb,
                                         temperature=temperature, max_iterations=max_iterations)
            except AgentLoopError as exc:
                logger.warning("ReAct 调用链降级：%s", exc)
                buffer.append({"type": "fallback", "message": "已使用基础评估模式完成分析"})
                result = self._fallback_evaluate(session, resume_text, target_job)

            for event in buffer:
                yield event
            session.final_result = {"task": (filename, target_job), "data": result}
            session.touch()
            yield {"type": "result", "data": result}
        except AppError as exc:
            yield {"type": "error", "message": exc.message}
        except Exception as exc:
            logger.error("Agent 流式评估异常 error=%s", exc, exc_info=True)
            yield {"type": "error", "message": f"服务内部错误：{exc}"}


# ============ 辅助函数 ============


def _temp_save(file_bytes: bytes, ext: str) -> str:
    """保存上传文件到临时目录，返回路径"""
    from app.config import UPLOAD_DIR

    from app.utils.common_tools import generate_id

    path = UPLOAD_DIR / f"{generate_id('tmp')}{ext}"
    path.write_bytes(file_bytes)
    return str(path)


def _to_score(value: Any) -> int:
    """安全转分数（0-100）"""
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _weighted_total(dimensions: dict) -> int:
    """按权重计算总分"""
    if not dimensions:
        return 0
    weight_sum = sum(SCORE_WEIGHTS.values()) or 1.0
    weighted = sum(SCORE_WEIGHTS.get(k, 0) * v for k, v in dimensions.items())
    return round(weighted / weight_sum)


def _auto_weaknesses(dimensions: dict) -> list:
    """根据低分维度自动生成弱点清单"""
    return [f"{k}得分偏低（{v}分）" for k, v in dimensions.items() if v < 80] or ["整体匹配度尚可，建议强化项目量化成果"]
