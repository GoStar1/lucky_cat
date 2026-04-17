import csv
import os

os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from FlagEmbedding import FlagModel
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)

CSV_FILE = os.path.join(os.path.dirname(__file__), "csv", "public.csv")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = "journals"
EMBED_DIM = 1024
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))


def load_csv() -> list[dict]:
    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            if cleaned.get("ID"):
                rows.append(cleaned)
    return rows


def build_text(row: dict) -> str:
    parts = [
        row.get("期刊名称（中文）", ""),
        row.get("期刊名称（英文）", ""),
        row.get("期刊类型", ""),
        row.get("标签", ""),
        row.get("征稿方向", ""),
    ]
    return " ".join(p for p in parts if p)


def ensure_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已清空旧 collection: {COLLECTION_NAME}")

    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema("name_cn", DataType.VARCHAR, max_length=256),
        FieldSchema("name_en", DataType.VARCHAR, max_length=256),
        FieldSchema("journal_type", DataType.VARCHAR, max_length=64),
        FieldSchema("tags", DataType.VARCHAR, max_length=512),
        FieldSchema("topics", DataType.VARCHAR, max_length=2048),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ])
    client.create_collection(COLLECTION_NAME, schema=schema)
    print(f"已创建 collection: {COLLECTION_NAME}")


def main():
    rows = load_csv()
    print(f"读取 CSV：{len(rows)} 条记录")

    model = FlagModel("BAAI/bge-large-zh-v1.5", use_fp16=True)
    print("模型加载完成")

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    ensure_collection(client)

    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [build_text(r) for r in batch]
        embeddings = model.encode(texts).tolist()

        data = [
            {
                "id": r.get("ID", ""),
                "name_cn": r.get("期刊名称（中文）", "")[:256],
                "name_en": r.get("期刊名称（英文）", "")[:256],
                "journal_type": r.get("期刊类型", "")[:64],
                "tags": r.get("标签", "")[:512],
                "topics": r.get("征稿方向", "")[:2048],
                "embedding": emb,
            }
            for r, emb in zip(batch, embeddings)
        ]
        client.insert(COLLECTION_NAME, data)
        total += len(batch)
        print(f"已插入 {total}/{len(rows)}")

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
