"""Generate the self-contained Colab notebook from reviewed local source files."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'ITAI2373-NewsBot-Midterm'
cells=[]
def md(text): cells.append(nb.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nb.v4.new_code_cell(text.strip()))

md('''# NewsBot Intelligence System
**Benjamin LaCount II · ITAI 2373 · Solo midterm · October 17, 2026**

[Repository](https://github.com/MarineMaaan/ITAI2373-NewsBot-Midterm) ·
[BBC dataset](https://www.kaggle.com/competitions/learn-ai-bbc/data)

## 1. Business context and run instructions
NewsBot helps a small media desk route articles, identify entities and relationships,
and flag tone for human review. The intended user is an editor or research analyst;
value comes from reducing repetitive sorting, with a human checking uncertain cases.
Benjamin's related interest is categorizing his studio's game reviews. That requires
new review data, a review-specific label scheme and separate validation; BBC news
accuracy does not establish review-analysis accuracy.

**Run in Google Colab free tier, CPU.** Runtime → Run all. The next setup cell installs
small libraries and the small English spaCy model. Download and extract **BBC News Train.csv**
from Kaggle yourself, then select that file in the explicit upload cell. No credentials,
automatic news download, dataset bundle, or sample-data fallback is used.
The supplied Test.csv is unlabeled and the sample solution is not ground truth.
Instructor approval for BBC was confirmed by the student.

BBC has **business, entertainment, politics, sport, tech**, not health. Do not infer
health support. Original case and punctuation are preserved for parsing; lemmas feed
the classifier. BBC text is largely lowercased, which weakens case-sensitive NER.

All eight modules and the four non-video bonus extensions are included. No team
name/number or video is used. The extra-credit list exceeds the stated 15-point cap;
the instructor determines the awarded total.''')
code('''import importlib.util, subprocess, sys
from pathlib import Path
IN_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IN_COLAB:
    packages = ['numpy>=1.26,<3', 'pandas>=2.2,<3', 'scikit-learn>=1.5,<1.8',
                'spacy>=3.8,<3.9', 'nltk>=3.9,<4', 'matplotlib>=3.9,<4',
                'seaborn>=0.13,<0.14', 'gradio>=5,<6']
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *packages])
import spacy, nltk
if not spacy.util.is_package('en_core_web_sm'):
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
assert nltk.download('vader_lexicon', quiet=True), 'NLTK resource download failed; check internet access.'
print('Setup complete. CPU only; no paid APIs or GPU required.')''')
md('''## Required dataset upload
Download the **labeled training CSV**, not kaggle.json, the test CSV or sample solution.
The notebook deliberately stops if no valid file is supplied. Local reproducibility
uses an explicitly supplied `NEWSBOT_LOCAL_CSV` path; Colab always uses the upload picker.''')
code('''import os
if IN_COLAB:
    from google.colab import files
    print('Upload BBC News Train.csv downloaded from Kaggle:')
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Upload exactly one labeled BBC training CSV.')
    filename, payload = next(iter(uploaded.items()))
    if not filename.lower().endswith('.csv'):
        raise ValueError('Please extract the download and upload the training CSV.')
else:
    local_path = os.environ.get('NEWSBOT_LOCAL_CSV')
    if not local_path:
        raise RuntimeError('Open in Colab to upload, or explicitly set NEWSBOT_LOCAL_CSV for local execution.')
    payload = Path(local_path).read_bytes()
print('File received; validation follows.')''')
md('''## Reusable implementation
These embedded modules make the notebook runnable by itself in Colab. The same
source is available as readable `.py` files in the repository.''')
code("Path('newsbot.py').write_text("+repr((PROJECT/'newsbot.py').read_text(encoding='utf-8'))+", encoding='utf-8')\nPath('extensions.py').write_text("+repr((PROJECT/'extensions.py').read_text(encoding='utf-8'))+", encoding='utf-8')\nprint('Reusable modules ready.')")
code('''import json, time, platform, importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score
from newsbot import *
from extensions import cross_lingual, train_custom_ner
START = time.perf_counter()
OUT = Path('outputs'); OUT.mkdir(exist_ok=True)
nlp = spacy.load('en_core_web_sm')
sia = SentimentIntensityAnalyzer()
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})
def savefig(name):
    plt.tight_layout(); plt.savefig(OUT / f'{name}.png', bbox_inches='tight'); plt.show(); plt.close()
df, audit = read_dataset(payload)
display(pd.Series(audit, name='Dataset audit').to_frame())
assert 500 <= len(df) <= 2000 and df.category.nunique() >= 4
df.category.value_counts().sort_index().plot.bar(color='#287d8e', figsize=(8,3))
plt.ylabel('Articles'); plt.title('Retained articles by category'); savefig('01_category_distribution')
df.assign(words=df.text.str.split().str.len()).boxplot(column='words', by='category', figsize=(8,3))
plt.suptitle(''); plt.title('Article length distribution'); plt.ylabel('Words'); savefig('02_article_lengths')''')
md('''## 2. Preprocessing and data quality
Clean HTML/entities, Unicode, URLs, email and whitespace. spaCy tokenizes and
lemmatizes, removes stop words and punctuation for lexical features. Original
normalized text remains available for POS, dependencies, NER and sentiment, so a
money amount or negation is not destroyed before those tasks. Exact canonical
duplicates are removed before splitting; near duplicates remain a possible risk.
English provenance comes from the selected BBC source plus a lightweight screen,
not a validated language detector. No fabricated titles, dates or sources are added.''')
code('''df = enrich(df,nlp,sia)
assert df.processed.str.len().gt(0).all()
splits = make_splits(df)
split_names = np.full(len(df),'',dtype=object)
for label,idx in zip(['train','validation','test'],splits): split_names[idx]=label
df['split']=split_names
display(df.groupby(['split','category']).size().unstack(fill_value=0))
before = df.text.str.split().str.len().mean(); after = df.processed.str.split().str.len().mean()
print(f'Mean tokens: before {before:.1f}; after {after:.1f}. Reduction {(1-after/before):.1%}.')
print(f'Preserved uppercase letter fraction: {audit["uppercase_letter_fraction"]:.3%}. Low case coverage limits NER.')
df[['row_id','category','split']].to_csv(OUT/'split_manifest.csv',index=False)''')
md('''## 3 & 7. TF-IDF, model comparison and optimization
Fit TF-IDF and chi-squared feature selection **inside each training fold**.
Three-fold stratified CV searches unigrams versus unigrams+bigrams, 75% versus
100% features, and two classifier settings for each of Naive Bayes, logistic
regression and linear SVM (24 configurations / 72 fold fits). Select the winner
by train CV macro-F1, before inspecting test performance. Train/validation/test
is approximately 70/15/15; validation sets a review threshold only.
This follows [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).''')
code('''models, cv_table, selected_name = fit_models(df,splits)
display(cv_table)
selected_model = models[selected_name]
threshold = review_threshold(df,splits,selected_model)
test_table, reports, predictions = score_models(df,splits,models)
display(test_table)
print('Selected by training CV:',selected_name)
display(pd.DataFrame(reports[selected_name]).T.round(4))
cv_table.to_csv(OUT/'cv_results.csv',index=False)
test_table.to_csv(OUT/'test_metrics.csv',index=False)
ytest = df.iloc[splits[2]].category.to_numpy()
ConfusionMatrixDisplay.from_predictions(ytest,predictions[selected_name],labels=CATEGORIES,xticks_rotation=30,cmap='Blues')
plt.title('Held-out test confusion matrix: CV-selected model'); savefig('03_confusion_matrix')
errors = pd.DataFrame({'row_id':df.iloc[splits[2]].row_id.to_numpy(),'actual':ytest,'predicted':predictions[selected_name]})
errors = errors[errors.actual != errors.predicted]
display(errors.head(12)); errors.to_csv(OUT/'classification_errors.csv',index=False)
test_table.set_index('model')[['accuracy','macro_f1']].plot.bar(ylim=(0,1),figsize=(8,3))
plt.title('Test results; model selection used training CV'); savefig('04_model_comparison')''')
code('''vectorizer = selected_model.named_steps['tfidf']
names = vectorizer.get_feature_names_out()
train = df.iloc[splits[0]]
matrix = vectorizer.transform(train.processed)
print(f'Training TF-IDF shape {matrix.shape}; sparsity {1-matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.2%}')
term_rows=[]
fig,axes=plt.subplots(1,len(CATEGORIES),figsize=(17,4))
for ax,category in zip(axes,CATEGORIES):
    mean=np.asarray(matrix[train.category.to_numpy()==category].mean(axis=0)).ravel()
    idx=mean.argsort()[-8:]
    ax.barh(names[idx],mean[idx],color='#287d8e'); ax.set_title(category)
    term_rows.extend({'category':category,'term':names[i],'mean_tfidf':float(mean[i])} for i in idx)
savefig('05_category_terms'); pd.DataFrame(term_rows).to_csv(OUT/'top_terms.csv',index=False)
selector=selected_model.named_steps['select']; kept=names[selector.get_support()]
classifier=selected_model.named_steps['model']
weights=classifier.coef_ if hasattr(classifier,'coef_') else classifier.feature_log_prob_
importance=[]
for label,row in zip(classifier.classes_,weights):
    for i in row.argsort()[-10:][::-1]: importance.append({'category':label,'feature':kept[i],'weight':float(row[i])})
display(pd.DataFrame(importance).groupby('category').head(5))
pd.DataFrame(importance).to_csv(OUT/'model_feature_weights.csv',index=False)
print('Category mean TF-IDF describes prevalence; signed coefficients or NB log-likelihoods describe model association. Neither is causal.')''')
md('''## 4. POS and grammatical patterns
Compare per-article POS proportions averaged within category, so long articles do
not dominate. POS bigrams describe local grammatical style. These are descriptive
corpus differences, not evidence of editorial quality or universal category rules.''')
code('''pos=pd.DataFrame(df.pos.tolist()).fillna(0); pos['category']=df.category
pos_summary=pos.groupby('category').mean()
major=[p for p in ['NOUN','VERB','ADJ','ADV','PROPN','NUM','PRON'] if p in pos_summary]
plt.figure(figsize=(9,4)); sns.heatmap(pos_summary[major].T,annot=True,fmt='.3f',cmap='YlGnBu')
plt.title('Mean article POS proportions'); savefig('06_pos_patterns')
pos_summary.to_csv(OUT/'pos_by_category.csv')
pattern_rows=[]
for label,group in df.groupby('category'):
    counts=Counter()
    for patterns in group.patterns: counts.update(patterns)
    pattern_rows.extend({'category':label,'pattern':p,'count':n} for p,n in counts.most_common(5))
display(pd.DataFrame(pattern_rows)); pd.DataFrame(pattern_rows).to_csv(OUT/'pos_bigrams.csv',index=False)''')
md('''## 5. Dependency parsing, semantic roles and syntax features
Extract subject/action/object relations, negation and passive constructions using
dependency labels. Agent and patient fields are **heuristic semantic roles**, not
a trained semantic-role labeling system. Coordination, indirect objects, reported
speech and missing agents are limitations. A missing role stays empty.
Sentence length, dependency distance, clause and passive rates are engineered
features; the research extension evaluates their classification contribution.''')
code('''syntax=pd.DataFrame(df.syntax.tolist()); syntax['category']=df.category
syntax_summary=syntax.groupby('category').mean()
display(syntax_summary.round(3)); syntax_summary.to_csv(OUT/'syntax_by_category.csv')
syntax_summary.plot.bar(subplots=True,layout=(2,2),figsize=(11,6),legend=False,rot=15)
savefig('07_syntax_features')
role_demo='The studio released a game. The game was reviewed by the editor. The developer did not cancel the launch.'
display(pd.DataFrame(doc_features(nlp(role_demo),sia)['relations']))''')
md('''## 6. Sentiment and emotional tone
[NLTK VADER](https://www.nltk.org/api/nltk.sentiment.vader.html) scores the original
text. Compound ≥0.05 is positive, ≤−0.05 negative, otherwise neutral. Scores reflect
language tone, not truth, endorsement or reader opinion. Four small, explicitly
listed lemma lexicons count joy, fear, anger and sadness cues; ties use declaration
order and zero matches become unmarked. This is an interpretable emotion proxy,
not validated emotion classification. BBC has no sentiment/emotion gold labels.''')
code('''sentiment_table=pd.crosstab(df.category,df.sentiment_label,normalize='index')
sentiment_table.plot.bar(stacked=True,figsize=(9,4),color=['#ba5d64','#a8afb8','#287d8e'])
plt.ylabel('Article proportion'); plt.title('Heuristic sentiment by category'); savefig('08_sentiment')
sentiment_table.to_csv(OUT/'sentiment_by_category.csv')
emotion_table=pd.crosstab(df.category,df.emotion,normalize='index')
display(emotion_table.round(3)); emotion_table.to_csv(OUT/'emotion_by_category.csv')
print('Emotion lexicons:',EMOTIONS)
print('Interpret category tone cautiously: reported deaths, sports losses and stock declines may trigger negative vocabulary without opinion.')''')
md('''## 8. Named entities and relationship mapping
Extract PERSON, ORG, GPE, DATE, MONEY with the pretrained English model. Aggregate
frequencies and category patterns. The network uses same-sentence co-occurrence;
an edge means the entities appeared together, not ownership, agreement or causation.
BBC's lowercasing especially limits PERSON/ORG/GPE detection. We do not repair case
by guessing names or claim NER accuracy without gold annotations.''')
code('''entity_rows=[]; edges=Counter()
for row in df.itertuples():
    entity_rows.extend({'category':row.category,'entity':e['text'].casefold(),'label':e['label']} for e in row.entities)
    for e in row.pairs: edges[(e['source'],e['target'])]+=e['count']
entities=pd.DataFrame(entity_rows,columns=['category','entity','label'])
entity_counts=entities.groupby(['label','entity']).size().rename('mentions').reset_index().sort_values('mentions',ascending=False)
display(entity_counts.groupby('label').head(5)); entity_counts.to_csv(OUT/'entity_frequencies.csv',index=False)
entity_cross=pd.crosstab(entities.category,entities.label).reindex(index=CATEGORIES,columns=sorted(ENTITY_LABELS),fill_value=0)
plt.figure(figsize=(8,4)); sns.heatmap(entity_cross,annot=True,fmt='d',cmap='Blues')
plt.title('Detected entity mentions, not verified facts'); savefig('09_entity_categories')
entity_cross.to_csv(OUT/'entities_by_category.csv')
edge_df=pd.DataFrame([{'source':a,'target':b,'sentence_count':int(n)} for (a,b),n in edges.most_common(20)])
edge_df.to_csv(OUT/'entity_relationships.csv',index=False)
display(edge_df.head(10))
top_edges=list(edges.most_common(12)); nodes=sorted({x for pair,_ in top_edges for x in pair})
if nodes:
    angles=np.linspace(0,2*np.pi,len(nodes),endpoint=False); coords={name:(np.cos(a),np.sin(a)) for name,a in zip(nodes,angles)}
    plt.figure(figsize=(10,8))
    for (a,b),weight in top_edges:
        x,y=coords[a],coords[b]; plt.plot([x[0],y[0]],[x[1],y[1]],alpha=.3,linewidth=1+np.log1p(weight),color='#287d8e')
    for name,(x,y) in coords.items(): plt.scatter(x,y,s=35,color='#193b59'); plt.text(1.08*x,1.08*y,name[:28],ha='center',fontsize=8)
    plt.axis('off'); plt.title('Entity relationship map: sentence co-occurrence'); savefig('10_entity_network')''')
md('''## Bonus: advanced cross-lingual comparison
Evaluate 25 original paired English/Spanish controlled passages, balanced across
the five labels. Compare English, Spanish directly, and Spanish with a fixed
50-word domain glossary. These short keyword-rich passages intentionally probe
language sensitivity. Their labels/pairs and glossary were specified before
scoring; a glossary matching the probe vocabulary makes this an optimistic
mechanism demonstration, **not general Spanish-language news evaluation**.
The main BBC model is never trained on these passages. Dates are absent from the
BBC CSV, so no publication chronology is invented.''')
code('''cross_pairs,cross_metrics=cross_lingual(selected_model,nlp,sia)
display(cross_metrics)
cross_pairs.to_csv(OUT/'crosslingual_pairs.csv',index=False); cross_metrics.to_csv(OUT/'crosslingual_metrics.csv',index=False)
cross_metrics.set_index('condition')[['accuracy','macro_f1']].plot.bar(ylim=(0,1),figsize=(8,3))
plt.title('Controlled bilingual probe, 25 pairs'); savefig('11_crosslingual')''')
md('''## Bonus: custom domain-specific NER training
Train a small spaCy transition-based NER model for **NEWS_OUTLET** and
**TECH_PRODUCT**, motivated by game-industry monitoring. The 96 train, 12 development
and 12 test sentences are original synthetic annotations. Name pairs **and sentence
frames** are disjoint between partitions; offsets are validated. Select the epoch
on development exact-span F1, then score the untouched test set once. Report
precision, recall and per-label F1 against a baseline emitting no custom labels.
This small controlled test does not measure BBC or real game-review NER performance.
[Training reference](https://spacy.io/usage/training).''')
code('''custom_ner,ner_metrics,ner_history,annotations,ner_predictions=train_custom_ner()
display(pd.Series({k:v for k,v in ner_metrics.items() if k != 'per_label'}).to_frame('Value'))
display(pd.DataFrame(ner_metrics['per_label']).T)
(OUT/'custom_ner_annotations.json').write_text(json.dumps(annotations,indent=2),encoding='utf-8')
(OUT/'custom_ner_test_predictions.json').write_text(json.dumps(ner_predictions,indent=2),encoding='utf-8')
ner_history.to_csv(OUT/'custom_ner_learning_curve.csv',index=False)
ner_history.plot(x='epoch',y='dev_f1',figsize=(8,3),legend=False)
plt.ylabel('Exact-span F1'); plt.title('Custom NER development learning curve'); savefig('12_custom_ner')''')
md('''## Bonus: research extension with ablation and uncertainty
**Question:** do grammar features add value to TF-IDF for BBC category routing?
The project-specific approach combines TF-IDF with normalized POS and dependency
statistics. Compare identical logistic regression (C=2) and unigram vocabulary,
without grammar versus with grammar, on the same untouched test partition.
Report 500 stratified paired-bootstrap resamples for the macro-F1 difference.
A second predeclared stress test masks words from detected entities while retaining
the original grammar measurements. This is a controlled lexical ablation, not an
end-to-end deployment simulation. Negative results are valid findings. This is an
original project experiment, not a claim of research-field novelty.''')
code('''research=research_experiment(df,splits)
display(pd.Series(research).to_frame('Result'))
ci=research['difference_ci95']
verdict='does not establish an improvement' if ci[0] <= 0 <= ci[1] else ('supports an improvement on this split' if ci[0] > 0 else 'suggests a decline on this split')
print(f'The paired interval {verdict}. Repeat across independent splits and corpora before generalizing.')''')
md('''## Integrated demonstration: two NEW authored articles
Both examples are original illustrative passages outside the training data. They
show the complete input → preprocessing → category → entities → sentiment → syntax
→ custom NER flow. Predictions are observations, not extra test-accuracy claims.
Low score margins trigger a human-review flag; the margin is not a calibrated
probability and cannot reliably detect out-of-domain text.''')
code('''new_articles=[
    'A football team won the championship after scoring twice in the final match. The coach praised the players and supporters celebrated at the stadium. The league will announce the next tournament schedule on Monday.',
    'Pixel Ledger reported that a game studio has released Aurora Engine, a software toolkit for computer graphics. Developers said the product improves rendering performance on laptops. The engineering team will publish a security update next month.'
]
demos=[analyze_article(t,selected_model,nlp,sia,threshold,custom_ner) for t in new_articles]
for i,result in enumerate(demos,1):
    print(f'NEW ARTICLE {i}'); display(pd.Series({k:result[k] for k in ['predicted_category','sentiment_label','emotion','review_required','entities','relations','custom_entities']}).to_frame('Result'))
(OUT/'new_article_demonstrations.json').write_text(json.dumps(demos,indent=2),encoding='utf-8')
# Quantify the validation-set review rule on the reserved test set.
test_records=df.iloc[splits[2]].to_dict('records')
scores=selected_model.decision_function(test_records) if hasattr(selected_model,'decision_function') else selected_model.predict_proba(test_records)
ordered=np.sort(scores,axis=1); accepted=(ordered[:,-1]-ordered[:,-2])>=threshold
selective={'test_coverage':float(accepted.mean()),'test_accuracy_when_accepted':float(accuracy_score(ytest[accepted],predictions[selected_name][accepted])) if accepted.any() else None,
           'validation_threshold':threshold,'caveat':'Threshold targets 80% validation coverage; coverage may shift. Margins are not probabilities.'}
display(pd.Series(selective).to_frame('Review policy'))''')
md('''## Bonus: interactive Gradio dashboard
Run this cell to inspect your own article. The interface calls the exact integrated
function used above. In Colab the temporary Gradio share link is needed for browser
access; it expires with the runtime. No hosting subscription or stored dataset is
required. Local execution binds only to loopback. Stop the cell/runtime when done.''')
code('''dashboard=create_dashboard(selected_model,nlp,sia,threshold,custom_ner)
if os.environ.get('NEWSBOT_SKIP_DASHBOARD') == '1':
    print('Automated run: dashboard constructed; launch is validated separately.')
else:
    dashboard.launch(share=IN_COLAB, inline=True, prevent_thread_lock=True)''')
md('''## Evidence, interpretation and business recommendations
The following summary is computed from this run. Main model selection uses training
CV only. Linguistic tables cover the retained corpus descriptively; their group
labels are never fed into classifier features. No sentiment/standard-NER accuracy
is claimed because the BBC CSV supplies only topic labels.''')
code('''winner=test_table.set_index('model').loc[selected_name]
high_num=pos_summary['NUM'].idxmax() if 'NUM' in pos_summary else 'unavailable'
tone=df.groupby('category').sentiment.mean().sort_values()
summary=f"""### Findings from this execution
- Retained **{len(df)}** unique substantial articles; split sizes **{[len(i) for i in splits]}**.
- Training CV selected **{selected_name}**. Reserved-test accuracy **{winner.accuracy:.3f}**, macro-F1 **{winner.macro_f1:.3f}**.
- **{high_num}** has the largest mean NUM proportion; structured amounts can help an analyst find quantity-heavy coverage.
- Mean heuristic tone ranges from **{tone.index[0]} ({tone.iloc[0]:.3f})** to **{tone.index[-1]} ({tone.iloc[-1]:.3f})**; this is lexical tone, not verified public opinion.
- Grammar ablation macro-F1: **{research['baseline_f1']:.3f} → {research['hybrid_f1']:.3f}**; difference CI **[{ci[0]:.3f}, {ci[1]:.3f}]**. It {verdict}.
- Custom NER exact-span test F1: **{ner_metrics['test_f1']:.3f}**, on 12 synthetic held-out sentences only.
- Human-review policy accepted **{selective['test_coverage']:.1%}** of held-out articles. Editorial time saved must be measured in a real workflow, not inferred from classifier accuracy.

### Recommendations
Use this as an assisted routing prototype. Review low-margin cases and audit errors by category.
Check entities against original articles; co-occurrence is only a lead for investigation.
For game reviews, collect licensed review text, define relevant labels (bugs, performance,
gameplay, value), annotate sentiment toward the game rather than whole-document tone,
and split by game/source/time to measure generalization. Begin with a small human-reviewed pilot.

### Limits and next experiments
BBC is a historical single-source benchmark. Its labels omit health, its CSV lacks
publication timestamps, and its casing limits NER. Exact deduplication does not remove
all near duplicates. Repeat grouped and temporal tests when metadata is available.
Sentiment/emotions and semantic roles need gold annotations. Bilingual probes and
custom NER data are synthetic and small. No production, fairness or real-time claim is made.
"""
display(Markdown(summary)); (OUT/'findings.md').write_text(summary,encoding='utf-8')
versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','spacy','nltk','matplotlib','gradio']}
results={'dataset':audit,'split_sizes':{k:len(v) for k,v in zip(['train','validation','test'],splits)},
         'selected_model':selected_name,'test_metrics':test_table.to_dict('records'),'classification_reports':reports,
         'research':research,'custom_ner':ner_metrics,'crosslingual':cross_metrics.to_dict('records'),
         'review_policy':selective,'versions':versions,'python':platform.python_version(),
         'elapsed_seconds':time.perf_counter()-START,'execution_context':'Local CPU validation; Colab upload path requires browser interaction.' if not IN_COLAB else 'Google Colab CPU'}
(OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
import joblib
Path('artifacts').mkdir(exist_ok=True)
joblib.dump({'model':selected_model,'threshold':threshold},'artifacts/runtime.joblib')
print('Outputs saved to:',OUT.resolve())
if IN_COLAB:
    import shutil
    shutil.make_archive('newsbot_outputs','zip',OUT)
    print('Use the next cell to download the result bundle.')''')
code('''if IN_COLAB:
    from google.colab import files
    files.download('newsbot_outputs.zip')
else:
    print('Local results remain in outputs/.')''')
md('''## Contributions, reflection and sources
**Solo student:** Benjamin LaCount II. Thought, planning and final quality assurance
remain the student's responsibilities. AI assisted execution, coding, debugging,
documentation and journal drafting. The student supplied the reflection themes:
preprocessing strongly affects final quality; preparation took longer than expected;
future preparation should be more efficient; game-review categorization is a
motivating studio use case. The editable two-page reflective journal accompanies
this notebook. No team members, team number or video are represented.

Sources: [BBC classification competition](https://www.kaggle.com/competitions/learn-ai-bbc/data);
[scikit-learn pipelines/leakage](https://scikit-learn.org/stable/common_pitfalls.html);
[spaCy linguistic features](https://spacy.io/usage/linguistic-features);
[spaCy training](https://spacy.io/usage/training);
[NLTK VADER API](https://www.nltk.org/api/nltk.sentiment.vader.html);
[VADER paper, Hutto & Gilbert 2014](https://ojs.aaai.org/index.php/ICWSM/article/view/14550);
[Gradio interface](https://www.gradio.app/docs/gradio/interface).
The original course starter notebook informed the module sequence. The emotion
lexicons, bilingual probes and custom NER annotations are original instructional
assets authored with AI assistance. Dataset rights remain with their owners.''')

notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'colab':{'name':'NewsBot_Intelligence_System.ipynb','provenance':[]}})
nb.validate(notebook)
nb.write(notebook,PROJECT/'NewsBot_Intelligence_System.ipynb')
print(f'Built {len(cells)} cells.')
