import csv
import os
import time

import resend
from dotenv import load_dotenv

load_dotenv()

CSV_FILE = os.path.join(os.path.dirname(__file__), "match_result.csv")
FROM_EMAIL = os.getenv("FROM_EMAIL", "hello@publishing001.com")
SEND_INTERVAL = float(os.getenv("SEND_INTERVAL", "1"))  # 每封间隔秒数


def send_email(to: str, subject: str, body: str) -> dict:
    resend.api_key = os.environ["RESEND_API_KEY"]
    html = "<br>".join(body.splitlines())
    return resend.Emails.send({
        "from": FROM_EMAIL,
        "to": [to],
        "subject": subject,
        "html": html,
        "text": body,
    })


def main():
    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    to_send = [r for r in rows if r.get("邮件标题", "").strip() and r.get("邮件内容", "").strip()]
    print(f"共 {len(rows)} 条记录，其中 {len(to_send)} 条有邮件内容，开始发送...\n")

    success, failed = 0, 0
    for i, row in enumerate(to_send, start=1):
        email = row["用户邮箱"].strip()
        subject = row["邮件标题"].strip()
        body = row["邮件内容"].strip()
        try:
            result = send_email(email, subject, body)
            print(f"[{i}/{len(to_send)}] 已发送 → {email}  ID: {result['id']}")
            success += 1
        except Exception as e:
            print(f"[{i}/{len(to_send)}] 发送失败 → {email}  错误: {e}")
            failed += 1
        if i < len(to_send):
            time.sleep(SEND_INTERVAL)

    print(f"\n完成：成功 {success} 封，失败 {failed} 封")


if __name__ == "__main__":
    main()
