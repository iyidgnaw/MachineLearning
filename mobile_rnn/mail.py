from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

def mail(file):
	msg = MIMEMultipart()
	att1 = MIMEText(open(file, 'rb').read(), 'base64', 'gb2312')
	att1["Content-Type"] = 'application/octet-stream'
	att1["Content-Disposition"] = 'attachment; filename="%s"'%file
	msg.attach(att1)

	msg['to']='monitorforwdylizk@gmail.com'
	msg['from'] = 'monitorforwdylizk@gmail.com'
	msg['subject'] = 'result'

	server = smtplib.SMTP('smtp.gmail.com:587')
	server.starttls()
	server.login('monitorforwdylizk@gmail.com','workincasia')
	server.sendmail(msg['from'], msg['to'],msg.as_string())
	server.quit()

