"""Exercise the running Gradio endpoint with original articles and bad input."""
from pathlib import Path
import json
from gradio_client import Client

root=Path(__file__).resolve().parents[1]
client=Client('http://127.0.0.1:7860',verbose=False)
assert '/predict' in client.view_api(return_format='dict',print_info=False)['named_endpoints']
cases=[('sport','A football team won the championship after its striker scored two goals in the final match. The coach praised the players and supporters celebrated the victory at the stadium. The league will announce its next tournament on Monday.'),
       ('tech','Pixel Ledger reported that a game studio has released Aurora Engine, a software toolkit for computer graphics. Developers said the product improves rendering performance on laptops. The engineering team will publish a security update next month.')]
results=[]
for expected,text in cases:
    summary,details=client.predict(text,api_name='/predict')
    assert details['predicted_category']==expected
    assert all(k in details for k in ['entities','sentiment','relations','emotions','custom_entities','review_required'])
    results.append({'expected_category':expected,'summary':summary,'details':details})
    print('PASS:',summary)
try:
    client.predict('',api_name='/predict')
except Exception as exc:
    assert 'eight words' in str(exc)
    print('PASS: blank input returns a readable validation error.')
else:
    raise AssertionError('Empty input was accepted by dashboard.')
(root/'ITAI2373-NewsBot-Midterm'/'outputs'/'dashboard_api_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
