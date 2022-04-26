from cryptography.fernet import Fernet
key = b'74fMvJErkn5C19YpSsKbM-WcNmJpuTVGSsWV0X1rxoo='
cipher_suite = Fernet(key)
ciphered_text = cipher_suite.encrypt(b'Haha')
with open('encrypt.bin', 'wb') as file_object:  file_object.write(ciphered_text)
