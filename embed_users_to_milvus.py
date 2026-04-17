import csv
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import yake
from FlagEmbedding import FlagModel
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)

CSV_FILE = os.path.join(os.path.dirname(__file__), "csv", "user.csv")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = "users"
EMBED_DIM = 1024
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))


def _extract_one(abstract: str) -> list[str]:
    """进程池 worker，每个进程独立初始化 extractor。"""
    extractor = yake.KeywordExtractor(lan="en", n=2, top=5, dedupLim=0.7)
    if not abstract.strip():
        return []
    return [kw for kw, _ in extractor.extract_keywords(abstract)]


def load_csv() -> list[dict]:
    """读取 user.csv，按邮箱去重合并；YAKE 提取并行化。"""
    raw: dict[str, dict] = {}
    abstract_index: list[tuple[str, str]] = []  # [(email, abstract), ...]

    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            email = cleaned.get("邮箱", "").strip()
            if not email:
                continue

            if email not in raw:
                raw[email] = {
                    "email": email,
                    "name": cleaned.get("姓名", ""),
                    "institution": cleaned.get("机构", ""),
                    "research_directions": set(),
                    "all_keywords": [],
                }

            entry = raw[email]
            if not entry["name"] and cleaned.get("姓名"):
                entry["name"] = cleaned["姓名"]
            if not entry["institution"] and cleaned.get("机构"):
                entry["institution"] = cleaned["机构"]

            rd = cleaned.get("研究方向", "").strip()
            if rd:
                entry["research_directions"].add(rd)

            abstract = cleaned.get("摘要", "").strip()
            if abstract:
                abstract_index.append((email, abstract))

    # 并行提取关键词
    print(f"并行提取关键词（{len(abstract_index)} 条摘要）...")
    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_extract_one, ab): email for email, ab in abstract_index}
        for future in as_completed(futures):
            email = futures[future]
            raw[email]["all_keywords"].extend(future.result())

    records = []
    for entry in raw.values():
        seen, seen_set = [], set()
        for kw in entry["all_keywords"]:
            kw_lower = kw.lower()
            if kw_lower not in seen_set:
                seen_set.add(kw_lower)
                seen.append(kw)
            if len(seen) >= 20:
                break

        records.append({
            "email": entry["email"],
            "name": entry["name"],
            "institution": entry["institution"],
            "research_direction": "; ".join(sorted(entry["research_directions"])),
            "keywords": "; ".join(seen),
        })

    return records


def build_text(record: dict) -> str:
    parts = [record.get("research_direction", ""), record.get("keywords", "")]
    return " ".join(p for p in parts if p)


def ensure_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已清空旧 collection: {COLLECTION_NAME}")

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


def main():
    print("正在读取并合并 user.csv ...")
    records = load_csv()
    print(f"原始记录去重后：{len(records)} 个用户")

    model = FlagModel("BAAI/bge-large-zh-v1.5", use_fp16=True)
    print("模型加载完成")

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    ensure_collection(client)

    total = 0
    # 用后台线程插入上一批，与编码下一批并行
    with ThreadPoolExecutor(max_workers=1) as insert_pool:
        pending = None
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i: i + BATCH_SIZE]
            texts = [build_text(r) for r in batch]
            embeddings = model.encode(texts).tolist()

            data = [
                {
                    "id": r["email"][:256],
                    "name": r["name"][:128],
                    "institution": r["institution"][:512],
                    "research_direction": r["research_direction"][:2048],
                    "keywords": r["keywords"][:1024],
                    "embedding": emb,
                }
                for r, emb in zip(batch, embeddings)
            ]

            # 等上一批插入完成，再提交新批
            if pending is not None:
                pending.result()
                total += pending_size
                print(f"已插入 {total}/{len(records)}")

            pending = insert_pool.submit(client.insert, COLLECTION_NAME, data)
            pending_size = len(batch)

        if pending is not None:
            pending.result()
            total += pending_size
            print(f"已插入 {total}/{len(records)}")

    client.flush(COLLECTION_NAME)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="IVF_FLAT",
        params={"nlist": 128},
    )
    client.create_index(COLLECTION_NAME, index_params)
    client.load_collection(COLLECTION_NAME)

    res = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"])
    print(f"完成，collection 共 {res[0]['count(*)']} 条向量")


if __name__ == "__main__":
    main()
