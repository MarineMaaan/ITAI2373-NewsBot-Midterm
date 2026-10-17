"""Launch the evaluated local dashboard after executing the notebook.

Only load the artifact you produced locally; pickle/joblib is not a safe format
for untrusted downloads. This file never fetches a model or dataset remotely.
"""
from pathlib import Path
import sys
import joblib
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

project=Path(__file__).resolve().parents[1]/'ITAI2373-NewsBot-Midterm'
sys.path.insert(0,str(project))
from newsbot import create_dashboard
artifact=project/'artifacts'/'runtime.joblib'
if not artifact.exists():
    raise SystemExit('Execute the notebook with your BBC training CSV first.')
runtime=joblib.load(artifact)
nlp=spacy.load('en_core_web_sm')
custom=spacy.load(project/'artifacts'/'custom_ner')
app=create_dashboard(runtime['model'],nlp,SentimentIntensityAnalyzer(),runtime['threshold'],custom)
app.launch(server_name='127.0.0.1',server_port=7860,share=False)
