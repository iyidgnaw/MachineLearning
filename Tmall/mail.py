from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

def mail(file):
	msg = MIMEMultipart()
	att1 = MIMEText(open(file, 'rb').read(), 'base64', 'gb2312')
	att1["Content-Type"] = 'application/octet-stream'
	att1["Content-Disposition"] = 'attachment; filename="result.txt"'
	msg.attach(att1)

	msg['to']='imwangdiyi@gmail.com'
	msg['from'] = 'imwangdiyi@gmail.com'
	msg['subject'] = 'result'

	server = smtplib.SMTP('smtp.gmail.com:587')
	server.starttls()
	server.login('imwangdiyi@gmail.com','521Zhongguo1314')
	server.sendmail(msg['from'], msg['to'],msg.as_string())
	server.quit()
