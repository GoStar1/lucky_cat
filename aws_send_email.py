import csv
import os
import time

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

CSV_FILE = os.path.join(os.path.dirname(__file__), "match_result.csv")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
FROM_EMAIL = os.getenv("FROM_EMAIL", "info@publishing001.com")
SEND_INTERVAL = float(os.getenv("SEND_INTERVAL", "1"))

client = boto3.client(
    'ses',
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def send_email(to: str, subject: str, body: str) -> str:
    html = "<br>".join(body.splitlines())
    response = client.send_email(
        Source=FROM_EMAIL,
        Destination={'ToAddresses': [to]},
        Message={
            'Subject': {'Data': subject, 'Charset': 'UTF-8'},
            'Body': {
                'Html': {'Data': html, 'Charset': 'UTF-8'},
                'Text': {'Data': body, 'Charset': 'UTF-8'},
            },
        },
    )
    return response['MessageId']


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
            message_id = send_email(email, subject, body)
            print(f"[{i}/{len(to_send)}] 已发送 → {email}  ID: {message_id}")
            success += 1
        except ClientError as e:
            print(f"[{i}/{len(to_send)}] 发送失败 → {email}  错误: {e.response['Error']['Message']}")
            failed += 1
        if i < len(to_send):
            time.sleep(SEND_INTERVAL)

    print(f"\n完成：成功 {success} 封，失败 {failed} 封")


if __name__ == "__main__":
    main()
