"""NL2SQL 安全防护 _clean_sql 单元测试

验证 SQL 注入拦截：
- 非 SELECT 语句拒绝
- DDL/DML 危险操作拒绝
- 多语句 / 注释 / 十六进制 / 危险函数拒绝
不依赖外部服务(用 object.__new__ 绕过 __init__ 避免连接 MySQL)。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules.rag_module.service import RagService
from app.utils.common_tools import AppError


@pytest.fixture()
def svc():
    """绕过 __init__ 实例化,不触发 MySQL/Chroma/Ollama 连接"""
    return object.__new__(RagService)


def test_valid_select(svc):
    """正常 SELECT 通过,并自动补 LIMIT"""
    sql = svc._clean_sql("SELECT * FROM job_info WHERE city='深圳'")
    assert sql.upper().startswith("SELECT")
    assert "LIMIT 50" in sql.upper()


def test_valid_select_with_limit(svc):
    """已有 LIMIT 不重复追加"""
    sql = svc._clean_sql("SELECT * FROM job_info LIMIT 10")
    assert sql.upper().count("LIMIT") == 1


def test_delete_rejected(svc):
    """DELETE 注入拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("DELETE FROM job_info")


def test_update_rejected(svc):
    """UPDATE 注入拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("UPDATE job_info SET salary_max=999")


def test_drop_table_rejected(svc):
    """DROP TABLE 注入拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("DROP TABLE job_info")


def test_insert_rejected(svc):
    """INSERT 注入拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("INSERT INTO job_info VALUES (1,2,3)")


def test_multi_statement_rejected(svc):
    """多语句(分号拼接)拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("SELECT * FROM job_info; DROP TABLE users")


def test_comment_rejected(svc):
    """SQL 注释拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("SELECT * FROM job_info -- comment")


def test_hex_literal_rejected(svc):
    """十六进制字面量拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("SELECT 0x61646d696e")


def test_dangerous_function_rejected(svc):
    """SLEEP/BENCHMARK 等危险函数拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("SELECT SLEEP(10)")


def test_union_still_single_table(svc):
    """UNION 查询(仍在 job_info 表内)允许——仅单表查询不算跨表注入"""
    sql = svc._clean_sql("SELECT * FROM job_info WHERE city LIKE '%深圳%' UNION SELECT * FROM job_info")
    assert "UNION" in sql.upper()


def test_not_select_rejected(svc):
    """非 SELECT 开头语句拦截"""
    with pytest.raises(AppError):
        svc._clean_sql("SHOW TABLES")


def test_describe_rejected(svc):
    """DESCRIBE 语句拦截(表结构探测)"""
    with pytest.raises(AppError):
        svc._clean_sql("DESCRIBE job_info")


def test_code_fence_stripped(svc):
    """markdown 代码块标记被剥离后正常执行"""
    sql = svc._clean_sql("```sql\nSELECT * FROM job_info```")
    assert sql.upper().startswith("SELECT")
