TODO：

python3 embed_to_milvus.py
python3 embed_users_to_milvus.py
python3 match_journals_to_users.py

# Lucky Cat

将期刊数据与用户数据导入 Milvus 向量数据库，并根据研究方向进行智能匹配推荐。

## 功能概览

| 脚本 | 功能 |
|------|------|
| `embed_to_milvus.py` | 将期刊数据（`public.csv`）向量化并导入 Milvus |
| `embed_users_to_milvus.py` | 将用户数据（`user.csv`）向量化并导入 Milvus |
| `match_journals_to_users.py` | 为每本期刊匹配最相关的用户，导出 CSV |

## 环境要求

- Python 3.10+
- Milvus 2.3+（本地或远程）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置 Milvus 连接

默认连接本地 `http://localhost:19530`，可通过环境变量修改：

```bash
export MILVUS_URI=http://your-milvus-host:19530
export MILVUS_TOKEN=your_token   # 无需认证时留空
```

### 2. 导入期刊数据

```bash
python embed_to_milvus.py
```

读取 `csv/public.csv`，写入 Milvus `journals` collection。

### 3. 导入用户数据

```bash
python embed_users_to_milvus.py
```

读取 `csv/user.csv`，按邮箱去重合并同一作者的多篇论文，用 YAKE 从摘要提取关键词后向量化，写入 Milvus `users` collection。

### 4. 期刊-用户匹配

```bash
python match_journals_to_users.py
```

为每本期刊找出 Top 5 最匹配的用户，结果写入 `match_result.csv`。

## 数据文件

| 文件 | 说明 | 记录数 |
|------|------|--------|
| `csv/public.csv` | 期刊数据，含名称、类型、征稿方向、标签等 | ~405 条 |
| `csv/user.csv` | 用户数据，含姓名、邮箱、机构、摘要、研究方向等 | ~10,000 条 |

## Milvus Collection 结构

**journals**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR | 期刊ID（主键） |
| name_cn | VARCHAR | 中文名称 |
| name_en | VARCHAR | 英文名称 |
| journal_type | VARCHAR | 期刊类型 |
| tags | VARCHAR | 标签 |
| topics | VARCHAR | 征稿方向 |
| embedding | FLOAT_VECTOR(1024) | 语义向量 |

**users**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR | 用户邮箱（主键） |
| name | VARCHAR | 姓名 |
| institution | VARCHAR | 所在机构 |
| research_direction | VARCHAR | 研究方向 |
| keywords | VARCHAR | 从摘要提取的关键词 |
| embedding | FLOAT_VECTOR(1024) | 语义向量 |

## 模型

使用 [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)，1024 维向量，首次运行时自动下载。

向量索引：IVF_FLAT，相似度度量：内积（IP）。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MILVUS_URI` | `http://localhost:19530` | Milvus 地址 |
| `MILVUS_TOKEN` | 空 | 认证 token |
| `BATCH_SIZE` | `128` | 向量化批次大小 |
| `TOP_K` | `5` | 每本期刊匹配的用户数 |
| `QUERY_BATCH` | `50` | 匹配时每批查询的期刊数 |



