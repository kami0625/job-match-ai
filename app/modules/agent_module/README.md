# 简历评分优化 Agent 模块说明

基于 **ReAct（Reasoning + Acting）** 模式的简历评分优化智能体，全本地 Ollama 推理。
Agent 内部自动调用本系统 RAG 接口获取岗位 JD，驱动多轮工具调用完成评估。

## 目录结构

```
app/modules/agent_module/
├── config.py      # Agent 专属配置：提示词模板、ReAct 调度参数、超时、打分权重
├── tools.py       # 工具层：LangChain 风格工具集合（AgentTool 封装）
├── memory.py      # 会话记忆：单轮上下文 + 中间打分结果缓存
├── parser.py      # ReAct 输出解析器：兼容 JSON/文本形式，容错解析
├── service.py     # 业务层：ReAct Agent 构建 / 执行入口 / 结果格式化 / 降级兜底
└── api.py         # 接口层：/api/agent/* 统一参数校验与响应封装
```

## 工具清单（tools.py）

| 工具名 | 功能 | 底层能力 |
| ---- | ---- | ---- |
| `tool_get_job_requirements` | 调用 RAG 系统 HTTP 接口获取目标岗位 JD 与知识库资料 | `requests` → `/api/rag/jobs`、`/api/rag/query`、`/api/rag/chat` |
| `tool_resume_parser` | 解析简历，结构化提取个人信息/技能/项目经历/教育经历 | 复用 `app/utils/doc_parser.py` + 本地 LLM 结构化 |
| `tool_calc_match_score` | 简历-JD 四维匹配打分（技能匹配/项目经验/学历要求/业务关键词） | 规则计算（稳定可控、可复现） |
| `tool_generate_suggestion` | 生成简历逐段优化建议与修改示例 | 本地 LLM + 失败兜底 |

## ReAct 调用流程

```
用户上传简历 + 目标岗位
        │
        ▼
┌─ ReAct 循环（最多 6 轮，整体 300s 超时）────────────────┐
│  1. LLM 思考 → 输出 {"thought", "action", "action_input"} │
│  2. 解析器校验动作（未知工具/格式错乱 → 降级）           │
│  3. 执行工具 → 得到 Observation → 回填上下文             │
│  4. 防重复调用：同一工具最多 2 次                         │
│  5. LLM 输出 final_answer → 结束                         │
└──────────────────────────────────────────────────────────┘
        │
        ▼
结果格式化：总分 / 四维得分 / 弱点清单 / 优化建议 / 修改示例
```

- **降级兜底**：ReAct 链路异常（超时/超轮数/解析失败/Ollama 故障）时，自动切换为编排式兜底路径，直接按流程调用工具，保证接口始终返回结构化结果（响应带 `fallback: true` 标记）。
- **记忆缓存**：会话内 `resume_struct`、`job_requirements`、`score`、`suggestion` 均缓存，重复请求不重复调用大模型。

## 配置项（.env 可覆盖）

| 变量 | 默认 | 说明 |
| ---- | ---- | ---- |
| `AGENT_MAX_ITERATIONS` | 6 | ReAct 最大工具调用轮数（防无限循环） |
| `AGENT_TEMPERATURE` | 0.2 | 推理温度（低温度保证打分稳定） |
| `AGENT_TIMEOUT` / `AGENT_MODEL_TIMEOUT` | 180 / 120 | 单轮 LLM 调用超时（秒） |
| `AGENT_MAX_ROUND_SECONDS` | 300 | 整个调用链最大耗时（秒） |
| `AGENT_WEIGHT_SKILL` | 0.35 | 技能匹配权重 |
| `AGENT_WEIGHT_PROJECT` | 0.25 | 项目经验权重 |
| `AGENT_WEIGHT_EDUCATION` | 0.15 | 学历要求权重 |
| `AGENT_WEIGHT_KEYWORD` | 0.25 | 业务关键词权重 |

## 接口调用示例

```bash
# 1. Agent 一键评估（简历文件 + 目标岗位）
curl -X POST http://127.0.0.1:8000/api/agent/evaluate \
  -F "file=@样例简历_Java开发工程师.txt" -F "target_job=Java开发工程师"

# 2. 流式评估（展示思考过程）
curl -N -X POST http://127.0.0.1:8000/api/agent/evaluate/stream \
  -F "file=@样例简历_Java开发工程师.txt" -F "target_job=Java开发工程师"

# 3. 清空会话
curl -X POST http://127.0.0.1:8000/api/agent/clear -H "Content-Type: application/json" \
  -d '{"all": true}'

# 4. 单独获取优化建议
curl -X POST http://127.0.0.1:8000/api/agent/suggestion -H "Content-Type: application/json" \
  -d '{"resume_text": "简历文本...", "target_job": "Java开发工程师"}'
```

## 演示素材

`data/demo/` 目录提供：
- `样例岗位数据.sql`：10 条岗位数据，导入 MySQL 即可使用
- `样例简历_Java开发工程师.txt`：可直接上传评估的简历
- `样例岗位JD_Java开发工程师.txt`：岗位 JD 参考

## 常见故障与处理

| 现象 | 处理方式 |
| ---- | ---- |
| 模型不调用工具直接输出 | 解析器将其视为 final_answer，若缺字段则自动补 weaknesses/suggestions |
| 重复调用同一工具 | 单工具 2 次上限保护，超限切换兜底编排 |
| 输出格式错乱 | 解析器兼容 JSON/文本/代码块，多次尝试失败后降级 |
| RAG 接口不可用/返回空 | 工具返回明确 message，Agent 转为通用评估，不中断 |
| Ollama 超时/未启动 | ollama_client 重试 2 次后抛统一异常，触发降级兜底 |
