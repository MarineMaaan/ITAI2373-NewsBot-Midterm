from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
project=root/'ITAI2373-NewsBot-Midterm'
r=json.loads((project/'outputs'/'results.json').read_text(encoding='utf-8'))
winner=next(x for x in r['test_metrics'] if x['model']==r['selected_model'])
rows='\n'.join(f"| {x['model']} | {x['accuracy']:.4f} | {x['macro_f1']:.4f} |" for x in r['test_metrics'])
cross='\n'.join(f"| {x['condition']} | {x['accuracy']:.3f} | {x['macro_f1']:.3f} |" for x in r['crosslingual'])
ci=r['research']['difference_ci95']
text=f'''# NewsBot Intelligence System

**Benjamin LaCount II · ITAI 2373 · solo midterm · October 17, 2026**

[Public GitHub repository](https://github.com/MarineMaaan/ITAI2373-NewsBot-Midterm) ·
[Run in Google Colab](https://colab.research.google.com/github/MarineMaaan/ITAI2373-NewsBot-Midterm/blob/main/ITAI2373-NewsBot-Midterm/NewsBot_Intelligence_System.ipynb) ·
[Assignment traceability](REQUIREMENTS_MATRIX.md) · [Validation](VALIDATION.md)

NewsBot routes news into five BBC topics, extracts entities and grammatical
relationships, analyzes lexical tone, and exposes the same pipeline through Gradio.
An editor or analyst can use it for assisted media monitoring and manually review
uncertain cases. A future adaptation could organize feedback about games produced
by the student's studio, after collecting and evaluating review-specific data.

## Run the notebook

1. Open the Colab link above; use a free CPU runtime.
2. Download the [BBC News Classification dataset](https://www.kaggle.com/competitions/learn-ai-bbc/data) yourself. Sign in to Kaggle and accept the applicable conditions if needed.
3. Extract the archive and locate **BBC News Train.csv**. The training file has `ArticleId`, `Text` and `Category` columns.
4. Choose **Runtime → Run all**. Setup installs the required libraries, the small English spaCy model and VADER data.
5. At the **required upload cell**, upload that one training CSV using the displayed picker. No Kaggle API token is required. The notebook has no automatic dataset download or example-data fallback.
6. Let all analysis cells finish. Use the Gradio article textbox for new English input. In Colab, a temporary share link provides access and expires with the runtime.
7. Download `newsbot_outputs.zip` from the final cell. Each rerun requires your uploaded input; Colab runtime resets remove uploaded files.

The notebook embeds the reusable Python modules, so it runs from a single `.ipynb`
without cloning this repository. Network access is required for dependency/model
setup, not for fetching the dataset. Do not upload `kaggle.json`, the unlabeled
competition test CSV, or the sample solution. A validation error explains invalid
inputs. BBC categories are **business, entertainment, politics, sport, tech**;
health is outside this model's label space. [Data provenance and checks](DATASET.md).

## Observed results

The fresh local CPU notebook run retained **{r['dataset']['retained_rows']}** articles,
removing **{r['dataset']['duplicates_removed']}** canonical duplicates. Partition sizes:
**{r['split_sizes']['train']} train / {r['split_sizes']['validation']} validation / {r['split_sizes']['test']} test**.
Three-fold training CV selected **{r['selected_model']}** before test evaluation.
TF-IDF vocabulary and feature selection are fitted inside folds to avoid leakage.

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
{rows}

![Held-out test confusion matrix](outputs/03_confusion_matrix.png)

The selected classifier achieves **{winner['accuracy']:.2%} accuracy** and
**{winner['macro_f1']:.4f} macro-F1** on this BBC split. These are topic metrics;
they do not measure sentiment, standard NER, real-time operation or game-review quality.
[Per-class reports, data hash and versions](outputs/results.json) ·
[Computed findings](outputs/findings.md) · [Classification errors](outputs/classification_errors.csv).

## Modules 1–8

| Module | What is implemented |
|---|---|
| Application context | Target editor/analyst, workflow, value and review-monitoring adaptation limits |
| Preprocessing | Markup/URL/Unicode cleaning, tokenization, stop words, lemmatization and input audits |
| TF-IDF | Unigram/bigram optimization, chi-squared selection, category terms and model feature weights |
| POS | Per-article POS proportions, cross-category heatmap and grammatical bigrams |
| Syntax/semantics | Dependency parsing, syntax features and heuristic agent/action/patient extraction |
| Sentiment/emotion | VADER tone and four transparent emotion cue lexicons with cross-category analysis |
| Classification | Naive Bayes, logistic regression and linear SVM; CV, per-class reports, confusion and errors |
| Named entities | PERSON, ORG, GPE, DATE, MONEY; frequencies, category patterns and relationship map |

![Category terms](outputs/05_category_terms.png)

Original normalized text is preserved for linguistic tasks; lexical classification
uses lemmas. BBC is largely lowercased, so preserved input cannot restore missing
case. NER outputs need verification. Agent/patient labels are dependency heuristics,
not a full trained semantic-role model. Entity graph edges mean same-sentence
co-occurrence, not a verified relationship. Sentiment and emotion have no gold
labels in this source and are not advertised as validated opinion measurements.

## All four non-video bonus opportunities

**Interactive dashboard (5 listed points).** Gradio uses the same `analyze_article`
function as the two new-article demonstrations. It returns category, tone, entities,
relations, custom entities and a margin-based review flag. Margins are not calibrated
probabilities. The threshold comes from validation, not test labels.

**Advanced analysis (5 listed points).** A controlled English/Spanish comparison
uses 25 original balanced passage pairs. Direct Spanish inference is compared with
a fixed glossary aid and English control. The glossary and probe terms overlap by
design: this is a language-sensitivity demonstration, not real Spanish news accuracy.

| Controlled probe | Accuracy | Macro-F1 |
|---|---:|---:|
{cross}

**Custom model (5 listed points).** A trained spaCy NER model learns `NEWS_OUTLET`
and `TECH_PRODUCT` from 96 authored synthetic sentences. Development and test each
have 12 sentences, with disjoint names **and frames**. Development selects the
checkpoint. Test exact-span precision **{r['custom_ner']['test_precision']:.3f}**,
recall **{r['custom_ner']['test_recall']:.3f}**, F1 **{r['custom_ner']['test_f1']:.3f}**;
the baseline emitting no custom labels scores F1 0.000. This small test does not
establish real-world domain NER quality. [Annotations](outputs/custom_ner_annotations.json)
and [held-out predictions](outputs/custom_ner_test_predictions.json) are included.

**Research extension (10 listed points).** The project-specific lexical+grammar
hybrid tests whether POS/dependency features improve a matched logistic regression
baseline. Baseline macro-F1 **{r['research']['baseline_f1']:.4f}** versus hybrid
**{r['research']['hybrid_f1']:.4f}**; hybrid-minus-baseline 95% paired bootstrap
interval **[{ci[0]:.4f}, {ci[1]:.4f}]**, using 500 stratified resamples.
The predeclared entity-word masking stress test also reports both systems.
One split and a conditional bootstrap do not prove universal improvement.
This is an original project investigation, not a claim of academic novelty.

The assignment states an extra-credit cap of 15 while these categories total 25.
All four categories are implemented; the instructor decides points. The optional
video bonus is excluded by the student's explicit requirement.

## Submission files

- [Executed, self-contained notebook](NewsBot_Intelligence_System.ipynb)
- [Two-page reflective journal, editable Word](documents/NewsBot_Reflection_Benjamin_LaCount_II.docx)
- [Two-page reflective journal, PDF](documents/NewsBot_Reflection_Benjamin_LaCount_II.pdf)
- [Technical report, editable Word](documents/MT_Report_Benjamin_LaCount_II_ITAI2373.docx)
- [Technical report, PDF](documents/MT_Report_Benjamin_LaCount_II_ITAI2373.pdf)
- [All figures and machine-readable results](outputs/)

Submit the public repository URL and reflective journal/report through the course's
Canvas submission. Canvas submission itself has not been performed. The report and
journal contain clickable repository links. Filenames identify the student and
omit group placeholders because this is solo work.

## Local reproducibility

From the repository root, using Python 3.11 or 3.12:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r ITAI2373-NewsBot-Midterm/requirements.txt
.\\.venv\\Scripts\\python.exe -m spacy download en_core_web_sm
```

To run the notebook locally, supply an explicit dataset path rather than relying on
any bundled data:

```python
import os, nbformat
from nbclient import NotebookClient
os.environ['NEWSBOT_LOCAL_CSV'] = r'C:\\path\\to\\BBC News Train.csv'
os.environ['NEWSBOT_SKIP_DASHBOARD'] = '1'
path = 'ITAI2373-NewsBot-Midterm/NewsBot_Intelligence_System.ipynb'
book = nbformat.read(path, as_version=4)
NotebookClient(book, timeout=1200, resources={{'metadata': {{'path': 'ITAI2373-NewsBot-Midterm'}}}}).execute()
nbformat.write(book, path)
```

Use an IPython kernel from the environment where you installed the dependencies.
On Windows this repository also provides `scripts/execute_notebook.py`, configured
for a manually downloaded `BBC News Train.csv` in the repository root; it creates
and removes a dedicated validation kernel. `scripts/check_core.py` and
`scripts/check_notebook.py` check behavior and executed-notebook/source parity.
After notebook execution, `scripts/run_dashboard.py` launches the dashboard at
`http://127.0.0.1:7860`. Its artifacts are generated locally, excluded from Git,
and should never be replaced with untrusted joblib files.

`requirements.txt` bounds supported direct dependencies; `environment-tested.txt`
records the actual local package set. Colab is supported by the notebook's setup
and upload cells. A signed-in Colab session has not been observed during validation.
Runtime recorded for the local analysis: **{r['elapsed_seconds']:.1f} seconds**.

## Specific contribution and AI disclosure

Benjamin LaCount II is the sole student contributor. He supplied the business
interest in studio game reviews and the reflection themes: preprocessing quality,
preparation efficiency, and the role of AI assistance. AI assisted execution,
implementation, debugging, evaluation and document drafting. Thought, planning,
understanding, final quality assurance and submission decisions remain the student's
responsibilities. The course starter informed the eight-module sequence. There
are no invented teammates, meetings, team names or team numbers.

## References

- [BBC News Classification competition and dataset](https://www.kaggle.com/competitions/learn-ai-bbc/data).
- [scikit-learn: leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html).
- [spaCy: linguistic features](https://spacy.io/usage/linguistic-features) and [training](https://spacy.io/usage/training).
- [NLTK VADER API](https://www.nltk.org/api/nltk.sentiment.vader.html).
- Hutto, C. J., & Gilbert, E. (2014). [VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text](https://ojs.aaai.org/index.php/ICWSM/article/view/14550). ICWSM.
- [Gradio Interface documentation](https://www.gradio.app/docs/gradio/interface).

The small emotion lexicons, paired probes and custom NER sentences are authored
instructional assets created with AI assistance. Original dataset rights remain
with their respective owners; no raw BBC CSV or Kaggle credential is published.
'''
(project/'README.md').write_text(text,encoding='utf-8')
print('Generated project README from observed results.')
