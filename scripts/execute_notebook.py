"""Execute the actual notebook in a fresh kernel with explicitly supplied data."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import json, sys, asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

root=Path(__file__).resolve().parents[1]
project=root/'ITAI2373-NewsBot-Midterm'
os.environ['NEWSBOT_LOCAL_CSV']=str(root/'BBC News Train.csv')
os.environ['NEWSBOT_SKIP_DASHBOARD']='1'
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
kernel_dir=root/'.qa'/'kernel'; kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'NewsBot validation','language':'python'}))
ksm=KernelSpecManager()
ksm.install_kernel_spec(str(kernel_dir),kernel_name='newsbot-validation',user=True)
path=project/'NewsBot_Intelligence_System.ipynb'
book=nbformat.read(path,as_version=4)
def cell_start(cell,cell_index,**kwargs):
    if cell.cell_type=='code': print(f'Executing code cell {cell_index+1}/{len(book.cells)}',flush=True)
client=NotebookClient(book,timeout=1200,kernel_name='newsbot-validation',resources={'metadata':{'path':str(project)}},on_cell_start=cell_start)
try:
    client.execute()
finally:
    nbformat.write(book,path)
    ksm.remove_kernel_spec('newsbot-validation')
print('Executed every notebook cell with no errors.')
