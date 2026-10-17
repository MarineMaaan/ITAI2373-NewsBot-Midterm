"""Create/verify only the requested owner's public repository using Git credentials.

Never prints, persists or embeds credential material. Push uses Git Credential
Manager independently. Re-running will inspect the existing repository safely.
"""
import json
import subprocess
import urllib.error
import urllib.request

OWNER='MarineMaaan'
NAME='ITAI2373-NewsBot-Midterm'
def request(path,method='GET',body=None):
    response=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,check=True)
    credentials=dict(line.split('=',1) for line in response.stdout.splitlines() if '=' in line)
    token=credentials.get('password')
    if not token: raise RuntimeError('No GitHub credential available through Git.')
    req=urllib.request.Request('https://api.github.com'+path,
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},
        data=json.dumps(body).encode() if body is not None else None,method=method)
    return json.load(urllib.request.urlopen(req))

if __name__=='__main__':
    user=request('/user')
    assert user['login']==OWNER, 'Authenticated owner differs from requested owner; refusing creation.'
    try:
        repo=request(f'/repos/{OWNER}/{NAME}')
        print('Repository already exists; verified owner:',repo['owner']['login'])
    except urllib.error.HTTPError as exc:
        if exc.code!=404: raise
        repo=request('/user/repos','POST',{'name':NAME,'private':False,'description':'Solo ITAI 2373 NewsBot midterm: Colab NLP pipeline, BBC upload, evaluated extensions and reflective journal.','auto_init':False})
        print('Created public repository.')
    assert repo['owner']['login']==OWNER and not repo['private']
    print(repo['html_url'])
