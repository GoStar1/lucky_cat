import csv
import hashlib
import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(sys.maxsize)

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from FlagEmbedding import FlagModel
from neo4j import GraphDatabase
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

CSV_FILE        = pathlib.Path(__file__).parent / "csv" / "user.csv"
DATA_DIR        = pathlib.Path(__file__).parent / "data" / "users"
COLLECTION_NAME = "users"
EMBED_DIM       = 1024
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "128"))
EMBED_MODEL     = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
MILVUS_URI      = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN    = os.getenv("MILVUS_TOKEN", "")
NEO4J_URI       = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER      = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")


# ── 文件系统 ────────────────────────────────────────────────────────────────

def user_file_path(email: str) -> pathlib.Path:
    h = hashlib.md5(email.encode()).hexdigest()
    return DATA_DIR / h[:2] / h[2:4] / f"{h}.json"


def update_user_file(row: dict) -> bool:
    email = row.get("邮箱", "").strip()
    if not email:
        return False

    path = user_file_path(email)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        user = json.loads(path.read_text(encoding="utf-8"))
        changed = False
    else:
        user = {"email": email, "name": "", "institution": "", "papers": [], "dirty": False}
        changed = True

    name        = row.get("姓名", "").strip()
    institution = row.get("机构", "").strip()
    if not user["name"] and name:
        user["name"] = name
        changed = True
    if not user["institution"] and institution:
        user["institution"] = institution
        changed = True

    paper = {
        "pmcid":              row.get("PMCID", "").strip(),
        "title":              row.get("标题", "").strip(),
        "abstract":           row.get("摘要", "").strip(),
        "research_direction": row.get("研究方向", "").strip(),
        "journal":            row.get("期刊", "").strip(),
        "publication_date":   row.get("发表时间", "").strip(),
    }

    # pmcid 优先，fallback title
    key_field = "pmcid" if paper["pmcid"] else "title"
    key_val   = paper[key_field]
    if key_val:
        existing = next(
            (i for i, p in enumerate(user["papers"]) if p.get(key_field) == key_val),
            None,
        )
        if existing is not None:
            if user["papers"][existing] != paper:
                user["papers"][existing] = paper
                changed = True
        else:
            user["papers"].append(paper)
            changed = True

    if changed:
        user["dirty"] = True
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    return changed


def load_dirty_users() -> list[dict]:
    result = []
    for p in DATA_DIR.rglob("*.json"):
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
            if user.get("dirty"):
                result.append(user)
        except (json.JSONDecodeError, OSError):
            pass
    return result


def mark_clean(batch_users: list[dict]):
    for user in batch_users:
        path = user_file_path(user["email"])
        user["dirty"] = False
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


# ── Embedding ───────────────────────────────────────────────────────────────

def build_user_record(user: dict) -> dict:
    directions = sorted({
        p["research_direction"]
        for p in user.get("papers", [])
        if p.get("research_direction")
    })
    return {
        "email":              user["email"],
        "name":               user.get("name", ""),
        "institution":        user.get("institution", ""),
        "research_direction": "; ".join(directions),
    }


def user_embed_texts(user: dict) -> list[str]:
    """收集该用户的所有嵌入文本：研究方向 + 各篇摘要（每条单独嵌入后取平均）。"""
    texts = []
    directions = sorted({
        p["research_direction"]
        for p in user.get("papers", [])
        if p.get("research_direction")
    })
    if directions:
        texts.append("; ".join(directions))
    for paper in user.get("papers", []):
        ab = paper.get("abstract", "").strip()
        if ab:
            texts.append(ab)
    return texts or [""]


# ── 数据库初始化 ────────────────────────────────────────────────────────────

def ensure_collection(client: MilvusClient) -> bool:
    """确保 collection 存在，返回是否是本次新建（新建则为空，无需 delete 旧数据）。"""
    expected = {"id", "email", "name", "institution", "research_direction", "embedding"}
    if client.has_collection(COLLECTION_NAME):
        info = client.describe_collection(COLLECTION_NAME)
        actual = {f["name"] for f in info["fields"]}
        if expected.issubset(actual):
            print(f"collection {COLLECTION_NAME} 已存在，跳过创建")
            client.load_collection(COLLECTION_NAME)
            return False
        print(f"collection {COLLECTION_NAME} 字段不匹配（paper-level 新 schema），drop 重建")
        client.drop_collection(COLLECTION_NAME)
    schema = CollectionSchema(fields=[
        FieldSchema("id",                 DataType.VARCHAR,      max_length=320, is_primary=True),
        FieldSchema("email",              DataType.VARCHAR,      max_length=256),
        FieldSchema("name",               DataType.VARCHAR,      max_length=128),
        FieldSchema("institution",        DataType.VARCHAR,      max_length=512),
        FieldSchema("research_direction", DataType.VARCHAR,      max_length=2048),
        FieldSchema("embedding",          DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ])
    client.create_collection(COLLECTION_NAME, schema=schema)
    print(f"已创建 collection: {COLLECTION_NAME}")
    return True


def ensure_index(client: MilvusClient):
    if "embedding" not in client.list_indexes(COLLECTION_NAME):
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="IP",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )
        client.create_index(COLLECTION_NAME, index_params)
        print("索引已创建")
    client.load_collection(COLLECTION_NAME)


def ensure_neo4j(driver):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT user_email_unique IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE")
        session.run("CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT journal_name_unique IF NOT EXISTS FOR (j:Journal) REQUIRE j.name IS UNIQUE")
    print("Neo4j 约束已就绪")


# ── 写库 ─────────────────────────────────────────────────────────────────────

def upsert_user_papers(client: MilvusClient, emails: list[str], entities: list[dict], skip_delete: bool = False):
    """paper-level 写入：先按 email 删除该批用户的旧记录，再插入新记录。"""
    if emails and not skip_delete:
        esc = [e.replace('"', '\\"') for e in emails]
        expr = 'email in [' + ",".join(f'"{e}"' for e in esc) + ']'
        client.delete(COLLECTION_NAME, filter=expr)
    if entities:
        client.insert(COLLECTION_NAME, entities)


def upsert_neo4j(session_factory, batch_records: list[dict], batch_users: list[dict]):
    users = [
        {
            "email":              r["email"],
            "name":               r["name"],
            "institution":        r["institution"],
            "research_direction": r["research_direction"],
        }
        for r in batch_records
    ]

    paper_rows = []
    for user in batch_users:
        for paper in user.get("papers", []):
            pid = paper.get("pmcid") or (
                hashlib.md5(paper["title"].encode()).hexdigest() if paper.get("title") else None
            )
            if not pid:
                continue
            paper_rows.append({
                "email":            user["email"],
                "paper_id":         pid,
                "title":            paper.get("title", ""),
                "publication_date": paper.get("publication_date", ""),
                "journal":          paper.get("journal", ""),
            })

    with session_factory() as session:
        session.run(
            "UNWIND $users AS u "
            "MERGE (node:User {email: u.email}) "
            "SET node.name = u.name, node.institution = u.institution, "
            "node.research_direction = u.research_direction",
            users=users,
        )
        session.run(
            "UNWIND $rows AS r "
            "MATCH (user:User {email: r.email}) "
            "MERGE (paper:Paper {id: r.paper_id}) "
            "  SET paper.title = r.title, paper.publication_date = r.publication_date "
            "MERGE (user)-[:AUTHORED]->(paper) "
            "WITH paper, r WHERE r.journal <> '' "
            "MERGE (journal:Journal {name: r.journal}) "
            "MERGE (paper)-[:PUBLISHED_IN]->(journal)",
            rows=paper_rows,
        )


def build_co_authored(driver):
    with driver.session() as session:
        session.run(
            "MATCH (u1:User)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(u2:User) "
            "WHERE u1.email < u2.email "
            "MERGE (u1)-[:CO_AUTHORED]-(u2)"
        )
    print("CO_AUTHORED 关系已建立")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    if not NEO4J_PASSWORD:
        raise RuntimeError("环境变量 NEO4J_PASSWORD 未设置")

    # Phase A: CSV → 文件系统
    print("正在读取 user.csv 并写入文件系统 ...")
    updated = 0
    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if update_user_file({k.strip(): v.strip() for k, v in row.items()}):
                updated += 1
    print(f"Phase A 完成：{updated} 条记录有变更")

    # Phase B: dirty → embedding → upsert
    dirty_users = load_dirty_users()
    print(f"需要更新的用户：{len(dirty_users)} 个")
    if not dirty_users:
        print("无需更新，退出。")
        return

    records = [build_user_record(u) for u in dirty_users]

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        neo4j_driver.verify_connectivity()
        print("Neo4j 连接成功")
        ensure_neo4j(neo4j_driver)

        model = FlagModel(EMBED_MODEL, use_fp16=True)
        print("模型加载完成")

        # 每条文本独立嵌入，paper-level 入库（BGE 输出已 L2 归一化）
        text_groups = [user_embed_texts(u) for u in dirty_users]
        flat_texts  = [t for group in text_groups for t in group]
        print(f"编码 {len(flat_texts)} 条文本（{len(dirty_users)} 个用户）...")
        flat_vecs = model.encode(flat_texts, batch_size=BATCH_SIZE)
        user_offsets: list[int] = []
        cursor = 0
        for group in text_groups:
            user_offsets.append(cursor)
            cursor += len(group)
        print("向量计算完成")

        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        fresh_collection = ensure_collection(client)

        total = 0
        with ThreadPoolExecutor(max_workers=1) as milvus_pool, \
             ThreadPoolExecutor(max_workers=1) as neo4j_pool:

            pending_milvus = pending_neo4j = None
            pending_size, pending_batch_users = 0, []

            for i in range(0, len(records), BATCH_SIZE):
                batch_records = records[i : i + BATCH_SIZE]
                batch_users   = dirty_users[i : i + BATCH_SIZE]

                batch_emails: list[str] = []
                milvus_data: list[dict] = []
                for j, (user, record) in enumerate(zip(batch_users, batch_records)):
                    user_idx = i + j
                    start    = user_offsets[user_idx]
                    n        = len(text_groups[user_idx])
                    email    = user["email"]
                    batch_emails.append(email)
                    for k in range(n):
                        milvus_data.append({
                            "id":                 f"{email}#{k}"[:320],
                            "email":              email[:256],
                            "name":               record["name"][:128],
                            "institution":        record["institution"][:512],
                            "research_direction": record["research_direction"][:2048],
                            "embedding":          flat_vecs[start + k].tolist(),
                        })

                if pending_milvus is not None:
                    pending_milvus.result()
                if pending_neo4j is not None:
                    pending_neo4j.result()
                if pending_batch_users:
                    mark_clean(pending_batch_users)
                    total += pending_size
                    print(f"已 upsert {total}/{len(records)} 用户")

                pending_milvus      = milvus_pool.submit(upsert_user_papers, client, batch_emails, milvus_data, fresh_collection)
                pending_neo4j       = neo4j_pool.submit(upsert_neo4j, neo4j_driver.session, batch_records, batch_users)
                pending_size        = len(batch_records)
                pending_batch_users = batch_users

            if pending_milvus is not None:
                pending_milvus.result()
            if pending_neo4j is not None:
                pending_neo4j.result()
            if pending_batch_users:
                mark_clean(pending_batch_users)
                total += pending_size
                print(f"已 upsert {total}/{len(records)} 用户")

        client.flush(COLLECTION_NAME)
        ensure_index(client)
        build_co_authored(neo4j_driver)

        res = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"])
        print(f"完成，Milvus collection 共 {res[0]['count(*)']} 条 paper-level 向量")

        with neo4j_driver.session() as session:
            n = session.run("MATCH (u:User) RETURN count(u) AS n").single()["n"]
        print(f"完成，Neo4j 共 {n} 个 User 节点")

    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    main()
