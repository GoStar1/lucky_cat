from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
EMAIL_THRESHOLD = float(os.getenv("EMAIL_THRESHOLD", "0.8"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "5"))

_COMPANY_INTRO_PATH = pathlib.Path(__file__).parent / "md" / "CompanyIntroduction.md"
COMPANY_INTRO = _COMPANY_INTRO_PATH.read_text(encoding="utf-8") if _COMPANY_INTRO_PATH.exists() else ""

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
- reason 字段的文字中绝对不能出现英文双引号（"），引用期刊或书名时请改用书名号《》。请写出匹配的论文题目和匹配的论文核心内容
- 只返回 JSON 数组，不要有任何额外文字或代码块标记。\
"""

_EMAIL_SYSTEM_PROMPT = f"""\
你是一位学术期刊编辑助理，负责向研究者发送期刊投稿邀请邮件。

以下是我们出版社的介绍，请在邮件中适当引用：
{COMPANY_INTRO}

邮件撰写要求：
1. 语言：根据研究者的论文摘要语言决定邮件语言——摘要以英文为主就写英文邮件，以中文为主就写中文邮件,摘要大部分是什么语言就写什么语言的邮件。
2. 语气：专业、友好、诚恳，避免过度推销。
3. 落款如何需要姓名的话就写编辑助理就好，翻译成邮件的语言就行。
4. 内容必须包含：
   - 对研究者具体研究方向的认可（体现个性化，结合其摘要内容）
   - 期刊与其研究方向的契合点（可引用匹配理由）
   - 出版社的简要优势介绍
   - 清晰的投稿邀请，并附上官网地址 www.usepress.org
5. 长度：正文 200–400 词（英文）或 150–300 字（中文）
6. 返回纯 JSON 数组，顺序与输入的研究者顺序一致，每个元素：
   {{"subject": "邮件标题", "body": "邮件正文"}}
7. body 中用 \\n 表示换行，绝对不能出现英文双引号（"），引用期刊名或书名请用《》。
8. 只返回 JSON 数组，不要有任何额外文字或代码块标记。\
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


def _llm_call_with_retry(client: anthropic.Anthropic, **kwargs) -> str:
    for attempt in range(3):
        try:
            resp = client.messages.create(**kwargs)
            return resp.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"\n  触发速率限制，等待 {wait}s 后重试...")
            time.sleep(wait)
    return ""


def _extract_json_array(raw: str) -> tuple[list | None, str]:
    """返回 (解析结果, 错误原因)。成功时错误原因为空字符串。"""
    m = re.search(r'\[[\s\S]*\]', raw)
    if not m:
        return None, "响应中未找到 [...] 结构（可能被截断）"
    try:
        result = json.loads(m.group(0))
        if isinstance(result, list):
            return result, ""
        return None, "JSON 根节点不是数组"
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败：{e}"


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
        snippet = abstract[:ABSTRACT_MAX_CHARS] if abstract else "（无摘要）"
        lines.append(
            f"研究者{idx}：\n"
            f"姓名：{entity['name']}  机构：{entity['institution']}\n"
            f"研究方向：{entity['research_direction']}\n"
            f"摘要节选：{snippet}"
        )

    raw = _llm_call_with_retry(
        client,
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[{"type": "text", "text": _LLM_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "\n\n".join(lines)}],
    )

    if not raw:
        return fallback

    results, err = _extract_json_array(raw)
    if results is None:
        print(f"\n  评分失败（{err}），片段：{raw[:200]}")
        return fallback
    if len(results) != len(ranked):
        print(f"\n  评分：返回条目数（{len(results)}）与期望（{len(ranked)}）不符")
        return fallback

    return [
        {
            "score": round(float(r.get("score", "")), 2) if r.get("score", "") != "" else "",
            "reason": r.get("reason", ""),
        }
        for r in results
    ]


def generate_emails_batch(
    journal: dict,
    high_scorers: list[dict],
    client: anthropic.Anthropic,
) -> list[dict]:
    fallback = [{"subject": "", "body": ""}] * len(high_scorers)

    lines = [f"期刊：{journal['name_cn']} / {journal['name_en']}\n"]
    for idx, hs in enumerate(high_scorers, start=1):
        abstract_snippet = hs["abstract"][:800] if hs["abstract"] else "（无摘要）"
        lines.append(
            f"研究者{idx}：\n"
            f"姓名：{hs['entity']['name']}  机构：{hs['entity']['institution']}\n"
            f"研究方向：{hs['entity']['research_direction']}\n"
            f"摘要节选：{abstract_snippet}\n"
            f"与本期刊匹配理由：{hs['reason']}"
        )

    raw = _llm_call_with_retry(
        client,
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[{"type": "text", "text": _EMAIL_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "\n\n".join(lines)}],
    )

    if not raw:
        return fallback

    results, err = _extract_json_array(raw)
    if results is None:
        print(f"\n  邮件生成失败（{err}），片段：{raw[:200]}")
        return fallback
    if len(results) != len(high_scorers):
        print(f"\n  邮件生成：返回条目数（{len(results)}）与期望（{len(high_scorers)}）不符")
        return fallback

    return [
        {
            "subject": r.get("subject", ""),
            "body": r.get("body", "").replace("\\n", "\n"),
        }
        for r in results
    ]


def _run_concurrent(tasks: list[tuple], worker_fn, total_label: str) -> dict:
    """并发执行任务，返回 {key: result} 字典。tasks 元素为 (key, *args)。"""
    results = {}
    done = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures = {executor.submit(worker_fn, *args): key for key, *args in tasks}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"\n  任务异常（{key}）：{e}")
                results[key] = None
            done += 1
            print(f"\r  {total_label}：{done}/{total}", end="", flush=True)
    print()
    return results


def main():
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    llm_client: anthropic.Anthropic | None = None
    if ENABLE_LLM_SCORING and ANTHROPIC_API_KEY:
        llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print(f"LLM 评分 + 邮件生成已启用（claude-sonnet-4-6，并发数 {LLM_CONCURRENCY}）")
        if not COMPANY_INTRO:
            print("警告：未找到 md/公司简介.md，邮件中将缺少出版社介绍")
    elif ENABLE_LLM_SCORING:
        print("警告：ENABLE_LLM_SCORING=true 但未设置 ANTHROPIC_API_KEY，跳过 LLM 功能")

    # ── 阶段 1：向量检索（快，串行即可）──────────────────────────────────
    print("读取期刊向量...")
    journals = fetch_all_journals(milvus_client)
    print(f"共 {len(journals)} 本期刊，开始向量检索...")

    abstract_cache: dict[str, str] = {}
    all_journal_ranked: list[tuple[dict, list]] = []  # [(journal, ranked)]

    for i in range(0, len(journals), QUERY_BATCH):
        batch = journals[i: i + QUERY_BATCH]
        search_results = milvus_client.search(
            "users",
            data=[j["embedding"] for j in batch],
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
            all_journal_ranked.append((journal, ranked))

    print(f"向量检索完成，共 {len(all_journal_ranked)} 本期刊")

    # ── 阶段 2：并发评分 ─────────────────────────────────────────────────
    score_map: dict[str, list[dict]] = {}
    if llm_client:
        tasks = [
            (journal["id"], journal, ranked, abstract_cache, llm_client)
            for journal, ranked in all_journal_ranked
        ]
        raw_results = _run_concurrent(tasks, score_with_llm, "评分进度")
        for journal, ranked in all_journal_ranked:
            jid = journal["id"]
            score_map[jid] = raw_results.get(jid) or [{"score": "", "reason": "评估失败"}] * len(ranked)
    else:
        for journal, ranked in all_journal_ranked:
            score_map[journal["id"]] = [{"score": "", "reason": ""}] * len(ranked)

    # ── 阶段 3：并发生成邮件（仅高分用户）──────────────────────────────
    email_map_all: dict[str, dict[str, dict]] = {}  # journal_id -> {user_email -> draft}
    if llm_client:
        email_tasks = []
        for journal, ranked in all_journal_ranked:
            llm_scores = score_map[journal["id"]]
            high_scorers = []
            for (email, (_, entity)), llm_result in zip(ranked, llm_scores):
                score_val = llm_result["score"]
                if isinstance(score_val, float) and score_val >= EMAIL_THRESHOLD:
                    high_scorers.append({
                        "email": email,
                        "entity": entity,
                        "score": score_val,
                        "reason": llm_result["reason"],
                        "abstract": load_user_abstracts(email, abstract_cache),
                    })
            if high_scorers:
                email_tasks.append((journal["id"], journal, high_scorers, llm_client))

        if email_tasks:
            def _email_worker(journal, high_scorers, client):
                drafts = generate_emails_batch(journal, high_scorers, client)
                return {hs["email"]: draft for hs, draft in zip(high_scorers, drafts)}

            raw_email = _run_concurrent(email_tasks, _email_worker, "邮件生成进度")
            for jid, email_map in raw_email.items():
                email_map_all[jid] = email_map or {}
        else:
            print("  没有匹配度 ≥ 0.8 的用户，跳过邮件生成")

    # ── 阶段 4：组装输出 ─────────────────────────────────────────────────
    output_rows = []
    for journal, ranked in all_journal_ranked:
        jid = journal["id"]
        llm_scores = score_map[jid]
        email_map = email_map_all.get(jid, {})
        for rank, ((email, (vec_score, entity)), llm_result) in enumerate(
            zip(ranked, llm_scores), start=1
        ):
            draft = email_map.get(email, {"subject": "", "body": ""})
            output_rows.append({
                "期刊ID": jid,
                "期刊名称（中文）": journal["name_cn"],
                "期刊名称（英文）": journal["name_en"],
                "排名": rank,
                "用户邮箱": email,
                "用户姓名": entity["name"],
                "所在机构": entity["institution"],
                "研究方向": entity["research_direction"],
                "相似度": round(vec_score, 4),
                "文章摘要": load_user_abstracts(email, abstract_cache),
                "llm匹配度": llm_result["score"],
                "llm匹配理由": llm_result["reason"],
                "邮件标题": draft["subject"],
                "邮件内容": draft["body"],
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"完成，共 {len(output_rows)} 条匹配结果，已导出到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
