import os, smtplib, ssl
from dotenv import load_dotenv

load_dotenv()

smtp_server = "smtp.gmail.com"
port = 465
sender_email = os.getenv("SMTP_EMAIL")
password = os.getenv("SMTP_PASSWORD")

print("DEBUG SMTP:", sender_email, bool(password))

message = """Subject: Test OTP Email

This is a test from smtp_test.py
"""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, sender_email, message)

print("Sent test email to:", sender_email)
