import csv
import hashlib
import json
import os
import pathlib
import re
import time

os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_TRACE", "")

from dotenv import load_dotenv
from pymilvus import MilvusClient
import anthropic

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
TOP_K = int(os.getenv("TOP_K", "5"))
OVERSAMPLE = int(os.getenv("OVERSAMPLE", "10"))
QUERY_BATCH = int(os.getenv("QUERY_BATCH", "50"))
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "match_result.csv")
DATA_DIR = pathlib.Path(__file__).parent / "data" / "users"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ENABLE_LLM_SCORING = os.getenv("ENABLE_LLM_SCORING", "true").lower() == "true"
ABSTRACT_MAX_CHARS = int(os.getenv("ABSTRACT_MAX_CHARS", "600"))

_LLM_SYSTEM_PROMPT = """\
你是一个学术期刊与研究者匹配评估专家。根据期刊定位与研究者的学术背景，客观评估每位研究者与该期刊的匹配程度。

评分标准（0.00–1.00）：
- 0.90–1.00：研究领域与期刊定位高度契合，发表概率很高
- 0.70–0.89：有明显重叠，存在合理的发表可能性
- 0.50–0.69：存在部分交叉，但核心方向有差距
- 0.30–0.49：关联较弱，匹配度低
- 0.00–0.29：几乎无关联

返回纯 JSON 数组，顺序与输入的研究者顺序一致，每个元素格式为：
{"score": 0.00, "reason": "中文1-2句说明匹配或不匹配的主要原因"}

重要约束：
- reason 字段的文字中绝对不能出现英文双引号（"），引用期刊或书名时请改用书名号《》。
- 只返回 JSON 数组，不要有任何额外文字或代码块标记。\
"""


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


def score_with_llm(
    journal: dict,
    ranked: list[tuple[str, tuple[float, dict]]],
    abstract_cache: dict[str, str],
    client: anthropic.Anthropic,
) -> list[dict]:
    fallback = [{"score": "", "reason": "评估失败"}] * len(ranked)

    lines = [f"期刊：{journal['name_cn']} / {journal['name_en']}\n"]
    for idx, (email, (_, entity)) in enumerate(ranked, start=1):
        abstract = load_user_abstracts(email, abstract_cache)
        abstract_snippet = abstract[:ABSTRACT_MAX_CHARS] if abstract else "（无摘要）"
        lines.append(
            f"研究者{idx}：\n"
            f"姓名：{entity['name']}  机构：{entity['institution']}\n"
            f"研究方向：{entity['research_direction']}\n"
            f"摘要节选：{abstract_snippet}"
        )
    user_text = "\n\n".join(lines)

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": _LLM_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_text}],
            )
            raw = resp.content[0].text.strip()
            # 用正则从响应中提取 JSON 数组，兼容 LLM 在数组前后加说明文字或 ``` 包裹的情况
            m = re.search(r'\[[\s\S]*\]', raw)
            if not m:
                print(f"  LLM 响应中未找到 JSON 数组，原始内容：{raw[:300]}")
                return fallback
            results = json.loads(m.group(0))
            if isinstance(results, list) and len(results) == len(ranked):
                return [
                    {
                        "score": round(float(r.get("score", "")), 2) if r.get("score", "") != "" else "",
                        "reason": r.get("reason", ""),
                    }
                    for r in results
                ]
            print(f"  LLM 返回条目数（{len(results)}）与期望（{len(ranked)}）不符")
            return fallback
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  触发速率限制，等待 {wait}s 后重试...")
            time.sleep(wait)
        except (anthropic.APIError, json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"  LLM 评分失败（{type(e).__name__}）：{e}")
            if 'raw' in dir():
                print(f"  原始响应片段：{raw[:300]}")
            return fallback

    return fallback


def main():
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    llm_client: anthropic.Anthropic | None = None
    if ENABLE_LLM_SCORING and ANTHROPIC_API_KEY:
        llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("LLM 评分已启用（claude-sonnet-4-6）")
    elif ENABLE_LLM_SCORING:
        print("警告：ENABLE_LLM_SCORING=true 但未设置 ANTHROPIC_API_KEY，跳过 LLM 评分")

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
            best: dict[str, tuple[float, dict]] = {}
            for hit in hits:
                email = hit["entity"]["email"]
                score = hit["distance"]
                if email not in best or score > best[email][0]:
                    best[email] = (score, hit["entity"])

            ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:TOP_K]

            if llm_client and ranked:
                llm_scores = score_with_llm(journal, ranked, abstract_cache, llm_client)
            else:
                llm_scores = [{"score": "", "reason": ""}] * len(ranked)

            for rank, ((email, (score, entity)), llm_result) in enumerate(
                zip(ranked, llm_scores), start=1
            ):
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
                    "llm匹配度": llm_result["score"],
                    "llm匹配理由": llm_result["reason"],
                })

        print(f"已处理 {min(i + QUERY_BATCH, len(journals))}/{len(journals)} 本期刊")

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"完成，共 {len(output_rows)} 条匹配结果，已导出到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
