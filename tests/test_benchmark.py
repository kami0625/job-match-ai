"""锚定评分 tool_benchmark_score 单元测试

验证评分区分度：
- 潦草简历必须 <80（不及格）
- 普通简历 60-79
- 良好简历 80-88
- 顶级简历 90+
不依赖 LLM / 外部服务（纯函数）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules.agent_module.tools import tool_benchmark_score, tool_resume_parser

JD = """【某公司 - Java开发工程师】 15-25K
要求:1.Java/SpringBoot/MyBatis/MySQL 2.Redis缓存 3.熟悉消息队列 4.本科及以上"""


def _score(resume_text: str) -> int:
    info = tool_resume_parser(resume_text=resume_text)
    return tool_benchmark_score(resume_info=info, job_requirements=JD, resume_text=resume_text)


# ---------- 各档简历样本 ----------

BRIEF = """测试简历_Java开发工程师张三
工作经历: 某公司 Java开发
负责Sass软件等产品的系统设计、核心开发与技术改造"""

SIMPLE = """个人简历 张三 本科 4年经验 Java开发
工作经历:某公司 Java开发工程师
- 负责订单模块开发,使用 SpringBoot+MySQL
专业技能:Java, SpringBoot, MySQL"""

FLUFF = """个人简历
姓名:张三
我是一名工作认真负责、吃苦耐劳、团队合作能力强的Java开发工程师。
我善于沟通,抗压能力强,学习能力强,积极向上,热爱学习。
我性格开朗,乐于助人,踏实肯干,具有良好的沟通能力。
工作经历:在某公司担任Java开发工程师,负责日常开发工作,参与项目开发。
我参与了公司多个项目的开发工作,包括订单系统、商品系统、用户系统等。
专业技能:掌握Java编程语言,了解Spring框架,了解MySQL数据库。"""

GOOD = """个人简历
姓名:张三
最高学历:本科
工作年限:4年
求职意向:Java开发工程师
教育:2016-2020 某理工大学 计算机本科
工作:2021至今 某金融科技公司 高级Java开发工程师
- 负责信贷核心订单模块,SpringBoot+MyBatis+MySQL,日均50万订单
- 主导状态机重构,Redis+分布式锁,800ms优化至120ms
- 参与微服务拆分,Spring Cloud Alibaba,12个微服务,双11平稳
- 搭建Kafka异步链路
2020-2021 某互联网公司 Java开发
- 参与商品后台系统,负责商品库存,单元测试70%+
- Docker容器化,自研Jenkins CI/CD
专业技能:Java 8/11/17/Spring/SpringBoot/MyBatis/MySQL/Redis/Kafka"""

EXCELLENT = """个人简历
姓名:李四
最高学历:硕士(985)
工作年限:8年
求职意向:Java架构师
教育:某985大学 计算机科学与技术 硕士
工作:某大厂 高级Java工程师/架构师
- 主导微服务架构升级,从单体演进到200+微服务,支撑日均千万级订单,系统可用性99.99%
- 设计分布式事务方案,双11峰值QPS 12万,订单成功率99.95%
- 搭建全链路压测平台,优化50+瓶颈点,接口响应时间提升8倍
- 主导Kafka集群建设,日处理200亿条消息,支撑30+业务线
- 团队管理:带领12人小组,主导2个开源项目
专业技能:Java/Spring/SpringBoot/Spring Cloud Alibaba/MyBatis/MySQL/Redis Cluster/Kafka/Docker/K8s/高并发/分布式/JVM调优/微服务架构"""


# ---------- 用例 ----------

def test_brief_resume_not_pass():
    """潦草简历必须不及格(<80)"""
    assert _score(BRIEF) < 80


def test_simple_resume_low_score():
    """简单简历(少量内容)分数低(<80)"""
    assert _score(SIMPLE) < 80


def test_fluff_resume_penalized():
    """废话套话简历即使字数多也不及格"""
    assert _score(FLUFF) < 80


def test_good_resume_pass():
    """良好简历(4年+多项目+量化)应达到及格线 80+"""
    assert _score(GOOD) >= 80


def test_excellent_resume_top():
    """顶级简历(985硕士+8年+多量化)应 90+"""
    assert _score(EXCELLENT) >= 90


def test_excellent_beats_good():
    """顶级简历分数必须高于良好简历(区分度)"""
    assert _score(EXCELLENT) > _score(GOOD)


def test_good_beats_brief():
    """良好简历分数必须高于潦草简历"""
    assert _score(GOOD) > _score(BRIEF)


def test_empty_resume_floor():
    """空简历分数不低于 15(封底)"""
    assert _score("") >= 15


def test_missing_skills_penalty():
    """技能与 JD 完全无关时加分受限(分数低于技能匹配简历)"""
    unrelated = """个人简历
姓名:王五
最高学历:本科
工作年限:3年
求职意向:UI设计师
教育:某美院 视觉传达 本科
工作:某设计公司 UI设计师
- 负责APP界面设计,Figma+Sketch,输出100+页面设计稿
- 主导设计规范搭建,复用率提升40%
专业技能:Figma/Sketch/Photoshop/AE"""
    assert _score(unrelated) < _score(GOOD)
