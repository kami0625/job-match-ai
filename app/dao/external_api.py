"""外部合规岗位数据源接入层

设计为「可插拔多数据源」(DataSource 模式):

- TencentCareersSource: 腾讯官网公开招聘,免费免 key,真实国内岗位(核心数据源)
- AdzunaSource: 国际岗位补充,需 app_id + app_key (5 分钟注册)
- 本地样例库(local): 覆盖字节/阿里/美团等多家公司代表性岗位,离线演示测试用
- 未来扩展:实现 ExternalJobSource 接口并注册到 SOURCE_REGISTRY 即可
  (企业级数据接入如猎聘/拉勾合作需企业资质,本版本留作扩展点)

约束:
1. 仅作为业务数据源接入,全程不涉及任何云端大模型 API (大模型推理仍走本地 Ollama);
2. 字段统一映射到本地 job_info 表结构,入库前按 岗位名+公司+城市 去重;
3. 所有数据源均选公开、免 Key 或条款允许聚合的合规接口;
4. 数据源启用条件: 配置文件就绪 (Adzuna 需 APP_ID/KEY,腾讯免 key 总可用)。
"""

import re
from abc import ABC, abstractmethod

import requests

from app.config import (
    ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRIES,
)
from app.utils.common_tools import AppError, get_logger

logger = get_logger("external_api")

class ExternalJobSource(ABC):
    """外部岗位数据源抽象接口"""

    name: str = ""
    display_name: str = ""
    description: str = ""

    def is_available(self) -> bool:
        """检查数据源是否就绪 (默认 True,需要鉴权的数据源 Override)"""
        return True

    @abstractmethod
    def fetch_jobs(self, limit: int = 30, **kwargs) -> list:
        """拉取原始岗位列表 (list[dict])"""

    @abstractmethod
    def to_job_info(self, raw: dict) -> dict:
        """字段映射 → job_info 表结构"""

    def safe_fetch(self, limit: int = 30, **kwargs) -> list:
        """安全拉取:不可用或失败时返回空列表,不抛异常"""
        if not self.is_available():
            logger.warning("[%s] 数据源不可用,跳过", self.name)
            return []
        try:
            return self.fetch_jobs(limit=limit, **kwargs)
        except AppError as exc:
            logger.warning("[%s] 拉取失败: %s", self.name, exc.message)
            return []
        except Exception as exc:
            logger.warning("[%s] 拉取异常: %s", self.name, exc)
            return []

# ============ Adzuna (主推,覆盖中国/英美等 8+ 国) ============

class AdzunaSource(ExternalJobSource):
    """Adzuna 公开 API:职位聚合,免费,8+ 国覆盖。需 app_id/app_key。

    注册: https://developer.adzuna.com/  (免费、5 分钟)
    文档: https://developer.adzuna.com/docs/search
    国家代码: gb / us / at / au / be / br / ca / ch / cn / de / es / fr / in / it / mx / nl / nz / pl / sg / za
    """

    name = "adzuna"
    display_name = "Adzuna 公开 API (多国聚合)"
    description = "免费聚合职位 API,覆盖中国/英美/印度等 8+ 国;需 app_id/app_key (5 分钟注册)"

    API_BASE = "https://api.adzuna.com/v1/api/jobs"
    TIMEOUT = 25

    def is_available(self) -> bool:
        return bool(ADZUNA_APP_ID and ADZUNA_APP_KEY)

    def fetch_jobs(
        self,
        limit: int = 30,
        what: str = "",
        where: str = "",
        salary_min: int = 0,
        country: str = "",
        **kwargs,
    ) -> list:
        """多国并行拉取 + 去重合并

        Args:
            limit: 最终返回总条数上限
            what: 关键词(岗位名/技能)
            where: 地点(城市/区域)
            salary_min: 最低薪资 (K/月,内部按国别换算后传入 Adzuna)
            country: 单一国家代码 (留空 = 配置的 ADZUNA_COUNTRIES 全查)
        """
        countries = [country] if country else list(ADZUNA_COUNTRIES)
        all_results, seen_ids = [], set()
        for cc in countries:
            batch_per_cc = max(5, limit // len(countries))  # 平均分配
            try:
                results = self._fetch_country(cc, batch_per_cc, what, where, salary_min)
            except AppError as exc:
                logger.warning("Adzuna %s 拉取失败: %s", cc, exc.message)
                continue
            for r in results:
                adzuna_id = f"{cc}_{r.get('id', '')}"
                if adzuna_id in seen_ids:
                    continue
                seen_ids.add(adzuna_id)
                r["_adzuna_country"] = cc
                r["_adzuna_id"] = adzuna_id
                all_results.append(r)
        logger.info(
            "Adzuna 多国拉取 共 %d 条 (countries=%s, what=%r, where=%r)",
            len(all_results), countries, what, where,
        )
        return all_results[:limit]

    def _fetch_country(
        self,
        country: str,
        limit: int,
        what: str,
        where: str,
        salary_min: int,
    ) -> list:
        """单国分页拉取"""
        results, page = [], 1
        while len(results) < limit and page <= 5:
            params = {
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "results_per_page": min(50, limit - len(results)),
                "what": what,
                "where": where,
                "content-type": "application/json",
            }
            if salary_min and salary_min > 0:
                params["salary_min"] = self._k_to_local(country, salary_min)

            url = f"{self.API_BASE}/{country}/search/{page}"
            try:
                resp = requests.get(url, params=params, timeout=self.TIMEOUT)
                if resp.status_code == 401:
                    raise AppError("Adzuna 鉴权失败: app_id/app_key 无效")
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                raise AppError(f"Adzuna {country} 网络失败: {exc}") from exc

            batch = data.get("results") or []
            if not batch:
                break
            results.extend(batch)
            page += 1
            count_total = data.get("count") or 0
            if count_total < 50:
                break
        return results[:limit]

    @staticmethod
    def _k_to_local(country: str, salary_min_k: int) -> int:
        """K/月 换算为 Adzuna salary_min 本币年度值

        cn (人民币): K/月 × 12000 = 元/年 (粗估, 实际公司发 14-16 薪)
        us/gb 等: K/月 × 1000 ~ 当地本币年度估算
        """
        if country == "cn":
            return int(salary_min_k * 12000)
        return int(salary_min_k * 1000)

    def to_job_info(self, raw: dict) -> dict:
        """Adzuna 字段 → job_info"""
        title = str(raw.get("title") or "").strip()

        comp_obj = raw.get("company") or {}
        company = str(comp_obj.get("display_name") or "").strip() if isinstance(comp_obj, dict) else ""

        loc_obj = raw.get("location") or {}
        city = str(loc_obj.get("display_name") or "").strip() if isinstance(loc_obj, dict) else ""
        country_tag = raw.get("_adzuna_country", "")
        if not city:
            city = _adzuna_country_to_default_city(country_tag)

        # 薪资归一化:cn/印度转 K/月,其它国家入库置 0 (前端显示「面议」)
        s_min = int(raw.get("salary_min") or 0)
        s_max = int(raw.get("salary_max") or 0)
        if country_tag == "cn" and (s_min or s_max):
            s_min = int(round(s_min / 12000)) if s_min else 0
            s_max = int(round(s_max / 12000)) if s_max else 0
        else:
            s_min, s_max = 0, 0

        # 描述: HTML 清洗 + 类目/链接
        desc = str(raw.get("description") or "").strip()
        desc = re.sub(r"<[^>]+>", " ", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        cat_obj = raw.get("category") or {}
        cat = str(cat_obj.get("label") or "").strip() if isinstance(cat_obj, dict) else ""

        parts = [desc[:600]] if desc else []
        if cat:
            parts.append(f"岗位类别: {cat}")
        if str(raw.get("salary_is_predicted")) == "1":
            parts.append("(薪资为 Adzuna 估算值,仅供参考)")
        parts.append(
            f"来源: Adzuna ({country_tag.upper()}) · "
            f"原始链接: {raw.get('redirect_url', '')}"
        )
        job_desc = "\n".join(parts)[:500]

        return {
            "job_name": title,
            "company": company,
            "city": city,
            "salary_min": s_min,
            "salary_max": s_max,
            "skill_require": cat,
            "education": "不限",
            "experience": _guess_experience(title),
            "job_desc": job_desc,
            "external_id": raw.get("_adzuna_id", ""),
            "source_name": self.name,
            "source_url": raw.get("redirect_url", ""),
        }

class TencentCareersSource(ExternalJobSource):
    """腾讯官网公开招聘接口(https://careers.tencent.com)

    完全合规:腾讯招聘页数据本身就是公开给求职者看的,访问其公开 JSON 接口属于"访问公开网页",
    不是爬取第三方平台,不违反 ToS。免费、无需 key、返回真实国内岗位。

    API: GET https://careers.tencent.com/tencentcareer/api/post/Query
      - pageIndex / pageSize 分页
      - keyword 关键词(岗位名/技能)
      - language=zh-cn&area=cn
    响应字段(实测):
      RecruitPostName 岗位名 / LocationName 城市 / BGName 事业群 / CategoryName 类别
      Responsibility 职责 / Requirement 要求 / RequireWorkYearsName 经验
      PostURL 岗位链接 / PostId / RecruitPostId / ComName 子公司
    """

    name = "tencent"
    display_name = "腾讯官网公开招聘（国内真实岗位）"
    description = "腾讯官网公开 JSON 接口,免费免 key,真实国内岗位(跨境电商/大模型/后端全覆盖)"

    API_URL = "https://careers.tencent.com/tencentcareer/api/post/Query"
    TIMEOUT = 25

    def fetch_jobs(self, limit=30, what="", where="", salary_min=0, **kwargs) -> list:
        """腾讯公开招聘搜索"""
        params = {
            "pageIndex": 1,
            "pageSize": min(50, limit),
            "language": "zh-cn",
            "area": "cn",
        }
        if what:
            params["keyword"] = what
        if where:
            params["keyword"] = f"{what} {where}" if what else where
        try:
            resp = requests.get(self.API_URL, params=params, timeout=self.TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if data.get("Code") != 200:
                raise AppError(f"腾讯招聘 API 返回异常: {data.get('Message', '')}")
            posts = (data.get("Data") or {}).get("Posts") or []
            logger.info("腾讯招聘拉取岗位 %d 条(keyword=%r)", len(posts), what)
            return posts[:limit]
        except requests.RequestException as exc:
            raise AppError(f"腾讯招聘 API 网络失败: {exc}") from exc

    def to_job_info(self, raw: dict) -> dict:
        """腾讯招聘字段 → job_info"""
        job_name = str(raw.get("RecruitPostName") or raw.get("PostName") or "").strip()
        company = str(raw.get("ComName") or "").strip()
        if not company:
            company = "腾讯"
        city = str(raw.get("LocationName") or "未知").strip()
        # 城市字段可能带后缀如 "深圳·南山",取主城市
        city = city.split("·")[0].split("-")[0].strip()

        # 腾讯招聘不公开薪资 → 置 0 前端显示"面议"
        skill = str(raw.get("CategoryName") or "").strip()
        # 经验
        exp_raw = str(raw.get("RequireWorkYearsName") or "").strip()
        experience = _guess_experience(exp_raw)
        if "经验" in exp_raw and exp_raw != "不限":
            experience = exp_raw

        desc_parts = []
        resp_text = str(raw.get("Responsibility") or "").strip()
        if resp_text:
            desc_parts.append("岗位职责:" + resp_text[:400])
        req_text = str(raw.get("Requirement") or "").strip()
        if req_text:
            desc_parts.append("任职要求:" + req_text[:400])
        if raw.get("BGName"):
            desc_parts.append("事业群:" + str(raw.get("BGName")))
        if raw.get("ProductName"):
            desc_parts.append("产品线:" + str(raw.get("ProductName")))
        desc_parts.append(
            "来源:腾讯官网公开招聘 · " + str(raw.get("PostURL", ""))
        )
        job_desc = chr(10).join(desc_parts)[:500]

        post_id = str(raw.get("PostId") or raw.get("RecruitPostId") or raw.get("Id") or "")
        return {
            "job_name": job_name,
            "company": company,
            "city": city,
            "salary_min": 0,   # 腾讯不公开薪资,前端显示面议
            "salary_max": 0,
            "skill_require": skill,
            "education": "不限",
            "experience": experience,
            "job_desc": job_desc,
            "external_id": f"tencent_{post_id}" if post_id else "",
            "source_name": self.name,
            "source_url": str(raw.get("PostURL") or ""),
        }

# ============ 数据源注册表 ============

SOURCE_REGISTRY: dict = {
    "adzuna": AdzunaSource(),
    "tencent": TencentCareersSource(),
}

def get_source(name: str) -> ExternalJobSource:
    """按标识获取数据源"""
    if name not in SOURCE_REGISTRY:
        raise AppError(
            f"不支持的数据源: {name},可选: {'、'.join(SOURCE_REGISTRY)}", code=400
        )
    return SOURCE_REGISTRY[name]

def list_sources() -> list:
    """列出全部数据源(含是否可用状态,前端选择用)"""
    return [
        {
            "name": src.name,
            "display_name": src.display_name,
            "description": src.description,
            "available": src.is_available(),
        }
        for src in SOURCE_REGISTRY.values()
    ]

def get_available_sources() -> list:
    """只列出可用的数据源"""
    return [s for s in list_sources() if s["available"]]

# ============ 工具函数 ============

def _guess_experience(title: str) -> str:
    """从岗位标题推断经验要求"""
    t = (title or "").lower()
    if any(k in t for k in ("senior", "lead", "principal", "staff", "architect", "manager", "高级", "资深", "架构师")):
        return "5年以上"
    if any(k in t for k in ("junior", "entry", "intern", "graduate", "trainee", "初级", "应届")):
        return "1-3年"
    if any(k in t for k in ("mid", "intermediate", "中级")):
        return "3-5年"
    return "不限"

def _adzuna_country_to_default_city(country: str) -> str:
    """无 location 信息时的兜底城市名"""
    mapping = {
        "cn": "中国",
        "gb": "英国",
        "us": "美国",
        "sg": "新加坡",
        "in": "印度",
        "de": "德国",
        "fr": "法国",
        "ca": "加拿大",
        "au": "澳大利亚",
    }
    return mapping.get((country or "").lower(), country.upper() if country else "海外")

# ============ 猎聘官方 MCP (国内数据最全 + 合规授权) ============

