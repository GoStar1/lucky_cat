import csv
import hashlib
import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import yake
from FlagEmbedding import FlagModel
from neo4j import GraphDatabase
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)

CSV_FILE   = pathlib.Path(__file__).parent / "csv" / "user.csv"
DATA_DIR   = pathlib.Path(__file__).parent / "data" / "users"
MILVUS_URI   = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = "users"
EMBED_DIM  = 1024
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ── 文件系统工具 ────────────────────────────────────────────────────────────

def user_file_path(email: str) -> pathlib.Path:
    h = hashlib.md5(email.encode()).hexdigest()
    return DATA_DIR / h[:2] / h[2:4] / f"{h}.json"


def update_user_file(row: dict) -> bool:
    """更新或创建用户文件，有变更则标记 dirty=True 并返回 True。"""
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
    """遍历 data/users/，返回所有 dirty=True 的用户。"""
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
    """将一批用户的 dirty 置 False 并原子写回。"""
    for user in batch_users:
        path = user_file_path(user["email"])
        user["dirty"] = False
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


# ── Embedding 工具 ──────────────────────────────────────────────────────────

def _extract_one(abstract: str) -> list[str]:
    extractor = yake.KeywordExtractor(lan="en", n=2, top=5, dedupLim=0.7)
    if not abstract.strip():
        return []
    return [kw for kw, _ in extractor.extract_keywords(abstract)]


def build_user_record(user: dict, keywords: list[str]) -> dict:
    directions = sorted({
        p["research_direction"]
        for p in user.get("papers", [])
        if p.get("research_direction")
    })
    seen, seen_set = [], set()
    for kw in keywords:
        kl = kw.lower()
        if kl not in seen_set:
            seen_set.add(kl)
            seen.append(kw)
        if len(seen) >= 20:
            break
    return {
        "email":              user["email"],
        "name":               user.get("name", ""),
        "institution":        user.get("institution", ""),
        "research_direction": "; ".join(directions),
        "keywords":           "; ".join(seen),
    }


def build_text(record: dict) -> str:
    parts = [record.get("research_direction", ""), record.get("keywords", "")]
    return " ".join(p for p in parts if p)


# ── 数据库初始化 ────────────────────────────────────────────────────────────

def ensure_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        print(f"collection {COLLECTION_NAME} 已存在，跳过创建")
        return
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema("name", DataType.VARCHAR, max_length=128),
        FieldSchema("institution", DataType.VARCHAR, max_length=512),
        FieldSchema("research_direction", DataType.VARCHAR, max_length=2048),
        FieldSchema("keywords", DataType.VARCHAR, max_length=1024),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ])
    client.create_collection(COLLECTION_NAME, schema=schema)
    print(f"已创建 collection: {COLLECTION_NAME}")


def ensure_index(client: MilvusClient):
    """建索引（幂等）并加载 collection。"""
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

def upsert_neo4j(session_factory, batch_records: list[dict], batch_users: list[dict]):
    users = [
        {
            "email":              r["email"],
            "name":               r["name"],
            "institution":        r["institution"],
            "research_direction": r["research_direction"],
            "keywords":           r["keywords"],
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
            "node.research_direction = u.research_direction, node.keywords = u.keywords",
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
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            if update_user_file(cleaned):
                updated += 1
    print(f"Phase A 完成：{updated} 条记录有变更")

    # Phase B: dirty → embedding → upsert
    dirty_users = load_dirty_users()
    print(f"需要更新的用户：{len(dirty_users)} 个")
    if not dirty_users:
        print("无需更新，退出。")
        return

    # 并行提取关键词
    abstract_index: list[tuple[int, str]] = []
    for idx, user in enumerate(dirty_users):
        for paper in user.get("papers", []):
            if paper.get("abstract", "").strip():
                abstract_index.append((idx, paper["abstract"]))

    print(f"并行提取关键词（{len(abstract_index)} 条摘要）...")
    user_keywords: dict[int, list[str]] = {i: [] for i in range(len(dirty_users))}
    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_extract_one, ab): idx for idx, ab in abstract_index}
        for future in as_completed(futures):
            user_keywords[futures[future]].extend(future.result())

    records = [
        build_user_record(user, user_keywords[i])
        for i, user in enumerate(dirty_users)
    ]

    # 连接数据库
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        neo4j_driver.verify_connectivity()
        print("Neo4j 连接成功")
        ensure_neo4j(neo4j_driver)

        model = FlagModel("BAAI/bge-large-zh-v1.5", use_fp16=True)
        print("模型加载完成")

        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        ensure_collection(client)

        total = 0

        with ThreadPoolExecutor(max_workers=1) as milvus_pool, \
             ThreadPoolExecutor(max_workers=1) as neo4j_pool:

            pending_milvus      = None
            pending_neo4j       = None
            pending_size        = 0
            pending_batch_users = []

            for i in range(0, len(records), BATCH_SIZE):
                batch_records = records[i : i + BATCH_SIZE]
                batch_users   = dirty_users[i : i + BATCH_SIZE]
                texts      = [build_text(r) for r in batch_records]
                embeddings = model.encode(texts).tolist()

                milvus_data = [
                    {
                        "id":                 r["email"][:256],
                        "name":               r["name"][:128],
                        "institution":        r["institution"][:512],
                        "research_direction": r["research_direction"][:2048],
                        "keywords":           r["keywords"][:1024],
                        "embedding":          emb,
                    }
                    for r, emb in zip(batch_records, embeddings)
                ]

                # 等上一批完成后再 mark_clean
                if pending_milvus is not None:
                    pending_milvus.result()
                if pending_neo4j is not None:
                    pending_neo4j.result()
                if pending_batch_users:
                    mark_clean(pending_batch_users)
                    total += pending_size
                    print(f"已 upsert {total}/{len(records)}")

                pending_milvus      = milvus_pool.submit(client.upsert, COLLECTION_NAME, milvus_data)
                pending_neo4j       = neo4j_pool.submit(upsert_neo4j, neo4j_driver.session, batch_records, batch_users)
                pending_size        = len(batch_records)
                pending_batch_users = batch_users

            # 收尾最后一批
            if pending_milvus is not None:
                pending_milvus.result()
            if pending_neo4j is not None:
                pending_neo4j.result()
            if pending_batch_users:
                mark_clean(pending_batch_users)
                total += pending_size
                print(f"已 upsert {total}/{len(records)}")

        client.flush(COLLECTION_NAME)
        ensure_index(client)
        build_co_authored(neo4j_driver)

        res = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"])
        print(f"完成，Milvus collection 共 {res[0]['count(*)']} 条向量")

        with neo4j_driver.session() as session:
            result = session.run("MATCH (u:User) RETURN count(u) AS n")
            n = result.single()["n"]
        print(f"完成，Neo4j 共 {n} 个 User 节点")

    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    main()
