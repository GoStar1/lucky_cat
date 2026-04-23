import boto3
from botocore.exceptions import ClientError

AWS_ACCESS_KEY_ID = 'AKIA6IH3CGJA77WQM3HZ'
AWS_SECRET_ACCESS_KEY = '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'
AWS_REGION = 'ap-southeast-1'

SENDER = 'info@publishing001.com'
RECIPIENT = 'hh13011315629@163.com'

client = boto3.client(
    'ses',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

try:
    response = client.send_email(
        Source=SENDER,
        Destination={
            'ToAddresses': [RECIPIENT]
        },
        Message={
            'Subject': {
                'Data': '2测试邮件',
                'Charset': 'UTF-8'
            },
            'Body': {
                'Html': {
                    'Data': '''
                    <html>
                    <body>
                        <h1>你好！</h1>
                        <p>这是一封来自 publishing001.com 的测试邮件。</p>
                        <p>如果你收到这封邮件，说明 AWS SES 配置成功。</p>
                        <br>
                        <p>-- Publishing001 团队</p>
                    </body>
                    </html>
                    ''',
                    'Charset': 'UTF-8'
                }
            }
        }
    )
    print('发送成功！MessageId:', response['MessageId'])

except ClientError as e:
    print('发送失败:', e.response['Error']['Message'])
