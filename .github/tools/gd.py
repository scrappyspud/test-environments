#!/usr/bin/env python3
# gd.py <drive link> <dest dir>: download a Drive file signed in as the service account in $GDRIVE_SA (its JSON key).
# The file has to be shared with the account's email. Prints nothing about the file but its size; exit 1 on failure.
import os, re, sys, json
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
url, dest = sys.argv[1], sys.argv[2]
m = re.search(r'(?:/d/|id=)([A-Za-z0-9_-]{20,})', url) or re.fullmatch(r'([A-Za-z0-9_-]{20,})', url.strip())
if not m: print('not a Drive link'); sys.exit(1)
fid = m.group(1)
cred = service_account.Credentials.from_service_account_info(json.loads(os.environ['GDRIVE_SA']), scopes=['https://www.googleapis.com/auth/drive.readonly'])
s = AuthorizedSession(cred)
meta = s.get(f'https://www.googleapis.com/drive/v3/files/{fid}', params={'fields': 'name,size', 'supportsAllDrives': 'true'})
if meta.status_code != 200:
    print(f'Drive said {meta.status_code}: ' + ('the file is not shared with the service account' if meta.status_code == 404 else meta.json().get('error', {}).get('message', '')[:200]))
    sys.exit(1)
name = os.path.basename(meta.json()['name']) or 'file.bin'
r = s.get(f'https://www.googleapis.com/drive/v3/files/{fid}', params={'alt': 'media', 'supportsAllDrives': 'true', 'acknowledgeAbuse': 'true'}, stream=True)
if r.status_code != 200: print(f'Drive said {r.status_code} on download'); sys.exit(1)
n = 0
with open(os.path.join(dest, name), 'wb') as f:
    for chunk in r.iter_content(8 << 20): f.write(chunk); n += len(chunk)
print(f'downloaded {n / 2**30:.2f} GB signed in')
