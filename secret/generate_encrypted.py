# %% import and definition
import base64
import os

import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

IN_DATA = "./data.csv"
IN_SEC = "./secret.yml"
IN_KEY = "./key"

# %% generating encrypted
with open(IN_KEY, "r") as keyf:
    pw = keyf.readline().encode("utf-8")
with open(IN_SEC, "r") as secf:
    sec = yaml.safe_load(secf)
with open(IN_DATA, "r") as datf:
    dat = "".join(datf.readlines())
salt = os.urandom(16)
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=480000,
)
key = base64.urlsafe_b64encode(kdf.derive(pw))
fernet = Fernet(key)
print("key salt: {}".format(salt))
print("encrypted app id: {}".format(fernet.encrypt(sec["id"].encode("utf-8"))))
print("encrypted app secret: {}".format(fernet.encrypt(sec["secret"].encode("utf-8"))))
print("encrypted data: {}".format(fernet.encrypt(dat.encode("utf-8"))))
