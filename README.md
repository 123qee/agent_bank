# 养老规划 Agent

任务 2 提交项目，核心入口为 `run.py`。

## 运行方式

```bash
python3 run.py "客户V500001现在年龄多大？"
```

标准输出只返回最终答案，例如：

```text
22岁
```

也支持评测程序直接导入：

```python
from run import run

answer = run("客户V500001每月可结余多少元？")
```

## 数据库配置

程序使用 `pymysql` 连接 MySQL，连接参数和表名都从环境变量读取：

- `TASK2_DB_HOST`
- `TASK2_DB_PORT`
- `TASK2_DB_USER`
- `TASK2_DB_PASSWORD`
- `TASK2_DB_NAME`
- `TASK2_BASE_TABLE`
- `TASK2_ACTION_TABLE`

本地默认使用训练表 `train_base_table` 和 `train_action_table`；正式评测会通过环境变量切换到测试表。

## 能力范围

- KYC 查询：年龄、性别、风险评级、净资产、收入、支出、月结余、退休金、企业年金。
- 聚合统计：年龄条件统计、权益类产品浏览客户平均年龄等。
- 行为偏好：通过 SQL 映射产品库类别并统计客户偏好。
- 养老测算：退休时间、退休时月支出、最低储备、可积累金额、缺口分析。
- 产品配置：收益最大化、最小化风险波动、长寿风险下的年金险配置。
- 建议书：生成包含 7 个章节的养老规划建议书。

## 提交说明

提交目录根目录必须包含 `run.py`。当前实现只依赖 Python 标准库和评测环境预装的 `pymysql`，不需要 `requirements.txt`。