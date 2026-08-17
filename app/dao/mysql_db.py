"""MySQL 连接与基础查询封装

基于 PyMySQL + DBUtils 连接池，提供：
1. job_info 岗位信息表自动建表
2. 通用查询（返回列表字典 / 首行字典）
3. 通用写操作（INSERT / UPDATE / DELETE / DDL）
4. 连接健康检查
"""

import threading
from typing import Any, Optional

import pymysql
from dbutils.pooled_db import PooledDB

from app.config import (
    MYSQL_CHARSET,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)
from app.utils.common_tools import AppError, get_logger, SQL_PCT_PLACEHOLDER

logger = get_logger("mysql_db")

# job_info 岗位信息表 DDL
# 注意：data_source + external_id 用于「外部数据源接入」架构,标识岗位来源 + 唯一去重
JOB_INFO_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS job_info (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '岗位ID',
    job_name VARCHAR(128) NOT NULL COMMENT '岗位名称',
    company VARCHAR(128) DEFAULT '' COMMENT '公司名称',
    city VARCHAR(64) DEFAULT '' COMMENT '所在城市',
    salary_min INT DEFAULT 0 COMMENT '薪资下限（K）',
    salary_max INT DEFAULT 0 COMMENT '薪资上限（K）',
    skill_require TEXT COMMENT '技能要求',
    education VARCHAR(64) DEFAULT '' COMMENT '学历要求',
    experience VARCHAR(64) DEFAULT '' COMMENT '经验要求',
    job_desc TEXT COMMENT '岗位描述',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '录入时间',
    data_source VARCHAR(16) NOT NULL DEFAULT 'local' COMMENT '数据来源: local / tencent / adzuna',
    external_id VARCHAR(64) NOT NULL DEFAULT '' COMMENT '外部数据源唯一 ID',
    source_url VARCHAR(512) DEFAULT '' COMMENT '原始申请链接 (外部数据源)',
    KEY idx_job_name (job_name),
    KEY idx_city (city),
    KEY idx_external_id (external_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='岗位信息表';
"""


def _add_column_if_not_exists(self_or_cur, table: str, col: str, ddl: str) -> None:
    """幂等: 给表加列(列已存在则跳过)。DDL 示例: 'VARCHAR(16) DEFAULT \"local\" COMMENT \"数据来源\"'"""
    # 用 self_or_cur.query_one 形式以兼容 single-instance 类的便捷调用
    try:
        exists = self_or_cur.query_one(
            "SELECT COUNT(*) AS cnt FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = %s",
            (MYSQL_DATABASE, table, col),
        ) or {"cnt": 0}
        if exists.get("cnt", 0) > 0:
            return
    except AppError:
        # information_schema 查询失败也尝试添加（由 MySQL 报错决定）
        pass
    self_or_cur.execute(f'ALTER TABLE `{table}` ADD COLUMN `{col}` {ddl}')


class MySQLDB:
    """MySQL 数据库访问封装（单例 + 连接池）"""

    _instance: Optional["MySQLDB"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MySQLDB":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._pool: Optional[PooledDB] = None

    def _get_pool(self) -> PooledDB:
        """懒加载连接池"""
        if self._pool is None:
            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=10,
                mincached=1,
                maxcached=5,
                blocking=True,
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                charset=MYSQL_CHARSET,
                cursorclass=pymysql.cursors.DictCursor,
            )
        return self._pool

    def query(self, sql: str, params: tuple = ()) -> list:
        """执行查询，返回全部行（列表字典）

        两层保护让 `%` 字符安全经过 PyMySQL：
        1. 还原 `SQL_PCT_PLACEHOLDER` 占位符为字面 `%`（service 层提前替换以避开 format 识别）
        2. params 为空时，把所有 `%` 转 `%%`（PyMySQL 会还原回 `%` 再下发给 MySQL），
           保证 LIKE 子句里的 `%` 通配符语义不受破坏（`%%` -> `%` 后 MySQL 仍按通配符解析）
        """
        safe_sql = sql.replace(SQL_PCT_PLACEHOLDER, "%")
        if not params:
            safe_sql = safe_sql.replace("%", "%%")
        conn = None
        try:
            conn = self._get_pool().connection()
            with conn.cursor() as cursor:
                cursor.execute(safe_sql, params)
                return list(cursor.fetchall() or [])
        except pymysql.MySQLError as exc:
            logger.error("MySQL 查询失败 sql=%s error=%s", sql[:200], exc)
            raise AppError(f"数据库查询失败：{exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    def query_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        """执行查询，返回首行字典，无结果返回 None"""
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple = ()) -> int:
        """执行写操作（INSERT/UPDATE/DELETE/DDL），返回受影响行数

        同样先还原占位符，params 为空时再 `%` -> `%%` 转义
        """
        safe_sql = sql.replace(SQL_PCT_PLACEHOLDER, "%")
        if not params:
            safe_sql = safe_sql.replace("%", "%%")
        conn = None
        try:
            conn = self._get_pool().connection()
            with conn.cursor() as cursor:
                cursor.execute(safe_sql, params)
                conn.commit()
                return cursor.rowcount
        except pymysql.MySQLError as exc:
            logger.error("MySQL 执行失败 sql=%s error=%s", sql[:200], exc)
            raise AppError(f"数据库操作失败：{exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    def insert_one(self, sql: str, params: tuple = ()) -> int:
        """执行插入，返回自增主键 ID

        同样先还原占位符，params 为空时再 `%` -> `%%` 转义
        """
        safe_sql = sql.replace(SQL_PCT_PLACEHOLDER, "%")
        if not params:
            safe_sql = safe_sql.replace("%", "%%")
        conn = None
        try:
            conn = self._get_pool().connection()
            with conn.cursor() as cursor:
                cursor.execute(safe_sql, params)
                conn.commit()
                return cursor.lastrowid or 0
        except pymysql.MySQLError as exc:
            logger.error("MySQL 插入失败 sql=%s error=%s", sql[:200], exc)
            raise AppError(f"数据库插入失败：{exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    def init_tables(self) -> None:
        """初始化建表（幂等）+ 兼容老表字段迁移"""
        self.execute(JOB_INFO_TABLE_SQL)
        # 老表兼容: 加 data_source + external_id（IF NOT EXISTS 替代品）
        try:
            _add_column_if_not_exists(
                self, "job_info", "data_source",
                'VARCHAR(16) NOT NULL DEFAULT \'local\' COMMENT \'数据来源: local / tencent / adzuna\''
            )
            _add_column_if_not_exists(
                self, "job_info", "external_id",
                'VARCHAR(64) NOT NULL DEFAULT \'\' COMMENT \'外部数据源唯一 ID\''
            )
            _add_column_if_not_exists(
                self, "job_info", "source_url",
                'VARCHAR(512) NOT NULL DEFAULT \'\' COMMENT \'外部数据源原始链接\''
            )
            # 老记录填默认 (data_source='local', external_id=job_name 以避免重复)
            self.execute(
                "UPDATE job_info SET data_source = 'local' "
                "WHERE data_source = '' OR data_source IS NULL"
            )
            # 已存在的 external_id 空字符串改为 job_name(避免唯一冲突,保证下次入库幂等)
            self.execute(
                "UPDATE job_info SET external_id = CONCAT('local_', id) "
                "WHERE external_id = ''"
            )
            # 索引(已存在则跳过,MySQL 没有 CREATE INDEX IF NOT EXISTS,手动查)
            try:
                existing_idx = self.query_one(
                    "SELECT COUNT(*) AS c FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA = DATABASE() "
                    "AND TABLE_NAME = 'job_info' AND INDEX_NAME = 'idx_external_id'"
                )
                if (existing_idx or {}).get("c", 0) == 0:
                    self.execute("CREATE INDEX idx_external_id ON job_info (external_id)")
            except AppError as exc:
                logger.warning("job_info 索引检查/创建跳过(可忽略): %s", exc.message)
        except AppError as exc:
            logger.warning("job_info 老表迁移失败(可忽略): %s", exc.message)
        logger.info("job_info 表初始化完成")

    def check_health(self) -> bool:
        """连接健康检查"""
        try:
            self.query_one("SELECT 1 AS ok")
            return True
        except AppError:
            return False
