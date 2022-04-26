import smtplib
import sys
from cryptography.fernet import Fernet
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

print("Sending Email")

# Create a text/plain message

sender = 'weppnesp@eckerd.edu'
recipent = sys.argv[1]
outer = MIMEMultipart()
outer['Subject'] = 'Completion of DWBA calclulation for '+sys.argv[1]
outer['From'] = sender
outer['To'] = recipent


composed = outer.as_string()

key = b'74fMvJErkn5C19YpSsKbM-WcNmJpuTVGSsWV0X1rxoo='
cipher_suite = Fernet(key)
with open('encrypt.bin', 'rb') as file_object:
    for line in file_object:
        encryptedpwd = line
uncipher_text = (cipher_suite.decrypt(encryptedpwd))
code = bytes(uncipher_text).decode("utf-8") #convert to string

# Send the message via our own SMTP server.
s = smtplib.SMTP('smtp.eckerd.edu', 587)
s.starttls()
s.login("weppnesp",code)
s.sendmail(sender, recipent, composed)
s.quit()

print("Email Sent")
