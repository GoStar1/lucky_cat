import csv
import hashlib
import json
import os
import pathlib

os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_TRACE", "")

from pymilvus import MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
TOP_K = int(os.getenv("TOP_K", "5"))
OVERSAMPLE = int(os.getenv("OVERSAMPLE", "10"))
QUERY_BATCH = int(os.getenv("QUERY_BATCH", "50"))
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "match_result.csv")
DATA_DIR = pathlib.Path(__file__).parent / "data" / "users"


def load_user_abstracts(email: str, cache: dict[str, str]) -> str:
    if email in cache:
        return cache[email]
    h = hashlib.md5(email.encode()).hexdigest()
    path = DATA_DIR / h[:2] / h[2:4] / f"{h}.json"
    abstracts: list[str] = []
    if path.exists():
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
            for paper in user.get("papers", []):
                ab = (paper.get("abstract") or "").strip()
                if ab:
                    abstracts.append(ab)
        except (json.JSONDecodeError, OSError):
            pass
    cache[email] = "\n\n".join(abstracts)
    return cache[email]


def fetch_all_journals(client: MilvusClient) -> list[dict]:
    results = []
    offset = 0
    while True:
        batch = client.query(
            "journals",
            filter="",
            output_fields=["id", "name_cn", "name_en", "embedding"],
            limit=1000,
            offset=offset,
        )
        results.extend(batch)
        if len(batch) < 1000:
            break
        offset += len(batch)
    return results


def main():
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    print("读取期刊向量...")
    journals = fetch_all_journals(client)
    print(f"共 {len(journals)} 本期刊")

    output_rows = []
    abstract_cache: dict[str, str] = {}

    for i in range(0, len(journals), QUERY_BATCH):
        batch = journals[i: i + QUERY_BATCH]
        embeddings = [j["embedding"] for j in batch]

        search_results = client.search(
            "users",
            data=embeddings,
            anns_field="embedding",
            limit=TOP_K * OVERSAMPLE,
            output_fields=["email", "name", "institution", "research_direction"],
        )

        for journal, hits in zip(batch, search_results):
            # 同一用户可能多次命中（paper-level 多向量），按 email 去重取 max 分数
            best: dict[str, tuple[float, dict]] = {}
            for hit in hits:
                email = hit["entity"]["email"]
                score = hit["distance"]
                if email not in best or score > best[email][0]:
                    best[email] = (score, hit["entity"])

            ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:TOP_K]
            for rank, (email, (score, entity)) in enumerate(ranked, start=1):
                output_rows.append({
                    "期刊ID": journal["id"],
                    "期刊名称（中文）": journal["name_cn"],
                    "期刊名称（英文）": journal["name_en"],
                    "排名": rank,
                    "用户邮箱": email,
                    "用户姓名": entity["name"],
                    "所在机构": entity["institution"],
                    "研究方向": entity["research_direction"],
                    "相似度": round(score, 4),
                    "文章摘要": load_user_abstracts(email, abstract_cache),
                })

        print(f"已处理 {min(i + QUERY_BATCH, len(journals))}/{len(journals)} 本期刊")

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"完成，共 {len(output_rows)} 条匹配结果，已导出到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
