# 基于RAG的求职匹配系统 技术实现规格说明书
## 一、项目概述
1.  项目定位：面向求职场景的混合数据源智能检索匹配系统，融合非结构化文档知识库与结构化岗位数据库，实现岗位查询、简历匹配、知识问答一体化能力。
2.  核心属性：全本地私有化部署，所有数据、大模型推理均在本地完成，不调用任何第三方云端大模型API。
3.  架构模式：前后端分离，FastAPI 提供后端推理服务接口，Streamlit 实现前端交互式页面。

## 二、整体技术栈
| 层级 | 技术选型 |
| ---- | ---- |
| 开发语言 | Python 3.10+ |
| 后端框架 | FastAPI + Uvicorn |
| 前端交互 | Streamlit |
| 大模型底座 | Ollama 本地部署，生成模型：qwen2:7b，嵌入模型：nomic-embed-text |
| 应用框架 | LangChain |
| 向量数据库 | Chroma |
| 关系型数据库 | MySQL 8.0 |
| 文档解析 | PyMuPDF（PDF）、python-docx（Word） |
| 检索增强 | BM25 关键词检索 + bge-reranker 精排 |
| 数据处理 | Pandas |

## 三、核心功能模块
### 1. 文档解析与入库模块
- 支持批量上传 PDF/Word 格式的岗位说明书、行业知识文档
- 自动完成文本提取、语义分块、向量化、元数据标注，存入 Chroma 向量库
- 支持增量更新，无需全量重建向量库

### 2. 混合检索 RAG 引擎模块
- 第一层召回：BM25 关键词检索 + 向量语义检索，双路召回结果合并去重
- 第二层精排：调用本地 bge-reranker 模型对召回片段做相似度重排序，取 Top K 结果
- 第三层查询改写：大模型自动对用户模糊提问进行扩写、补全，提升召回准确率
- 最终拼接检索上下文 + 用户问题，传入本地大模型生成回答，标注引用来源

### 3. NL2SQL 结构化查询模块
- 接收用户自然语言提问，调用本地大模型生成标准 MySQL 查询语句
- 自动执行 SQL，从岗位数据表中拉取符合条件的岗位数据
- 支持多条件筛选：城市、薪资、技能、学历、工作经验
- 内置 SQL 语法校验，避免非法语句执行

### 4. 岗位-简历匹配度计算模块
- 上传简历文本，自动提取核心技能、项目经验、学历等信息
- 与目标岗位 JD 做多维度相似度匹配，输出总分 + 分项得分
- 生成匹配度可视化图表（柱状图），基于 Matplotlib 实现

### 5. FastAPI 后端接口层
- 统一 RESTful 风格接口，做参数校验、异常处理、结果封装
- 所有大模型调用均走本地 Ollama 11434 端口，兼容 OpenAI 接口格式

### 6. Streamlit 前端交互层
- 侧边栏功能导航，分「文档管理」「智能问答」「岗位查询」「简历匹配」四个板块
- 支持文件上传、对话流式输出、结果表格展示、图表渲染

## 四、FastAPI 接口规范
### 1. 健康检查接口
- 路径：`GET /api/health`
- 返回：服务状态、Ollama 连接状态、向量库连接状态

### 2. 文档上传入库接口
- 路径：`POST /api/document/upload`
- 请求：multipart/form-data，字段 `file`（支持 pdf/docx）
- 返回：文档ID、入库状态、分块数量

### 3. RAG 智能问答接口
- 路径：`POST /api/rag/chat`
- 请求体：`{"query": "用户问题", "top_k": 3}`
- 返回：`{"answer": "生成回答", "sources": ["引用片段1", "引用片段2"]}`

### 4. NL2SQL 查询接口
- 路径：`POST /api/nl2sql/query`
- 请求体：`{"query": "自然语言查询条件"}`
- 返回：`{"sql": "生成的SQL语句", "result": [数据列表], "count": 结果条数}`

### 5. 简历匹配计算接口
- 路径：`POST /api/match/calculate`
- 请求体：`{"resume_text": "简历文本", "job_id": "目标岗位ID"}`
- 返回：`{"total_score": 85, "dimensions": {"技能匹配":90, "经验匹配":80}, "description": "匹配说明"}`

## 五、数据层设计
### 1. Chroma 向量库
- 集合名称：`job_knowledge_base`
- 元数据字段：doc_id、file_name、chunk_id、page_num、create_time

### 2. MySQL 岗位数据表（job_info）
| 字段名 | 类型 | 说明 |
| ---- | ---- | ---- |
| id | int 主键自增 | 岗位ID |
| job_name | varchar | 岗位名称 |
| company | varchar | 公司名称 |
| city | varchar | 所在城市 |
| salary_min | int | 薪资下限（K） |
| salary_max | int | 薪资上限（K） |
| skill_require | text | 技能要求 |
| education | varchar | 学历要求 |
| experience | varchar | 经验要求 |
| job_desc | text | 岗位描述 |
| create_time | datetime | 录入时间 |

## 六、代码工程约束
1.  分层架构：`app/api`（接口层）、`app/service`（业务逻辑）、`app/dao`（数据访问）、`app/config`（配置项）、`app/utils`（工具函数）
2.  配置文件统一管理：Ollama 地址、模型名称、数据库地址、向量库路径全部写入 `config.py`，支持一键切换
3.  全程无任何云端大模型 API Key、第三方在线接口调用，所有推理本地完成
4.  关键函数添加注释，异常捕获完善，避免服务崩溃