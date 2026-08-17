-- ============================================================
-- 求职匹配 AI 系统 - 演示岗位数据（10 条）
-- 用法：mysql -uroot -p -e "USE job_match_ai; SOURCE data/demo/样例岗位数据.sql;"
-- 说明：本文件自包含建表语句，可独立于后端启动先导入数据
-- ============================================================

USE job_match_ai;

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
    KEY idx_job_name (job_name),
    KEY idx_city (city)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='岗位信息表';

INSERT INTO job_info (job_name, company, city, salary_min, salary_max, skill_require, education, experience, job_desc) VALUES
('Java开发工程师', '星辰科技有限公司', '上海', 15, 25, 'Java, SpringBoot, MySQL, Redis, 分布式, 微服务', '本科', '3-5年', '负责公司核心业务系统的设计与开发，参与微服务架构改造，保障系统高可用与性能优化。'),
('Python后端开发工程师', '蓝鲸智能科技', '北京', 18, 30, 'Python, Django, Flask, FastAPI, PostgreSQL, Redis, Docker', '本科', '3-5年', '负责 AI 平台后端服务开发，设计高并发数据接口，维护容器化部署环境。'),
('前端开发工程师', '云帆互联网', '深圳', 13, 22, 'JavaScript, TypeScript, Vue, React, Webpack, 小程序', '本科', '1-3年', '负责 Web 端与小程序端开发，参与组件库建设与前端工程化改造，提升页面性能。'),
('数据分析师', '数智未来数据科技', '杭州', 12, 20, 'Python, SQL, Pandas, 数据分析, 数据可视化, ETL', '本科', '1-3年', '负责业务数据指标体系搭建，输出经营分析报告，支持业务决策与增长分析。'),
('算法工程师', '智算云科技', '北京', 25, 45, 'Python, 机器学习, 深度学习, PyTorch, TensorFlow, NLP', '硕士', '3-5年', '负责推荐算法与 NLP 模型研发，优化线上推理性能，推动算法效果持续提升。'),
('测试开发工程师', '品质护航科技', '上海', 12, 20, 'Python, 自动化测试, pytest, CI/CD, 接口测试', '本科', '1-3年', '负责自动化测试框架建设与 CI 流水线集成，保障产品发布质量。'),
('产品经理', '创新领航科技', '广州', 15, 25, '产品设计, 需求分析, 项目管理, Axure, 用户研究', '本科', '3-5年', '负责 ToB 产品规划与需求管理，协同研发团队推动产品迭代落地。'),
('运维开发工程师', '基石云科技', '深圳', 16, 26, 'Linux, Docker, Kubernetes, CI/CD, Shell, 监控', '本科', '3-5年', '负责云平台基础设施运维与自动化平台开发，提升系统稳定性与交付效率。'),
('大数据开发工程师', '数据星河科技', '杭州', 20, 32, 'Hadoop, Spark, Flink, Hive, Kafka, SQL', '本科', '3-5年', '负责离线与实时数仓建设，开发大数据处理任务，保障数据质量与时效。'),
('iOS开发工程师', '移动未来科技', '北京', 15, 24, 'Swift, Objective-C, iOS, 性能优化, 音视频', '本科', '3-5年', '负责 iOS 客户端核心功能开发与性能优化，参与架构升级与模块化改造。');
