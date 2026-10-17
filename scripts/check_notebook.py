"""Verify notebook structure, embedded-source parity and real upload-cell behavior."""
from pathlib import Path
import ast
import json
import sys
import types
import nbformat

root=Path(__file__).resolve().parents[1]
project=root/'ITAI2373-NewsBot-Midterm'
book=nbformat.read(project/'NewsBot_Intelligence_System.ipynb',as_version=4)
nbformat.validate(book)
code=[c for c in book.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code), 'Some code cells were not executed.'
assert not [o for c in code for o in c.outputs if o.output_type=='error']
embedded=next(c.source for c in code if "Path('newsbot.py').write_text" in c.source)
tree=ast.parse(embedded)
for filename in ['newsbot.py','extensions.py']:
    literal=next(n.value.args[0].value for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='write_text'
                 and n.value.func.value.args[0].value==filename)
    assert literal==(project/filename).read_text(encoding='utf-8'), f'Embedded {filename} differs from source.'

upload=next(c.source for c in code if 'uploaded = files.upload()' in c.source)
calls=[]
colab=types.ModuleType('google.colab')
colab.files=types.SimpleNamespace(upload=lambda:(calls.append('upload') or {'BBC News Train.csv':b'Text,Category\n'}))
google=types.ModuleType('google'); google.colab=colab
saved={k:sys.modules.get(k) for k in ['google','google.colab']}
try:
    sys.modules['google']=google; sys.modules['google.colab']=colab
    namespace={'IN_COLAB':True}
    exec(upload,namespace)
    assert calls==['upload'] and namespace['payload']==b'Text,Category\n'
    for supplied in [{},{'a.csv':b'x','b.csv':b'x'},{'kaggle.json':b'{}'}]:
        colab.files.upload=lambda supplied=supplied:supplied
        try: exec(upload,{'IN_COLAB':True})
        except ValueError: pass
        else: raise AssertionError('Invalid upload accepted.')
finally:
    for key,value in saved.items():
        if value is None: sys.modules.pop(key,None)
        else: sys.modules[key]=value
print(f'PASS: {len(code)} executed code cells; embedded source parity; required Colab upload and error branches.')
print('Colab upload was tested with a controlled API stub; this is not a signed-in Colab session.')
