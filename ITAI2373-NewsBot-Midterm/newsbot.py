"""NewsBot: bounded, reproducible news intelligence for BBC's five labels.

All learned transforms belong inside CV pipelines. Original text is kept for
linguistic analysis; cleaned lemmas feed the classifier. No dataset is downloaded.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']
ENTITY_LABELS = {'PERSON', 'ORG', 'GPE', 'DATE', 'MONEY'}
# Transparent instructional lexicon: coverage counts, not validated emotion probabilities.
EMOTIONS = {
    'joy': {'win', 'victory', 'celebrate', 'happy', 'success', 'delight', 'proud'},
    'fear': {'fear', 'threat', 'risk', 'danger', 'worry', 'crisis', 'terror'},
    'anger': {'anger', 'angry', 'protest', 'outrage', 'furious', 'dispute'},
    'sadness': {'death', 'die', 'loss', 'sad', 'tragedy', 'grief', 'mourn'},
}


def clean_text(value):
    """Normalize markup/Unicode/URLs without destroying casing, dates or money."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    text = unicodedata.normalize('NFKC', html.unescape(str(value)))
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+|\b\S+@\S+\.\S+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def read_dataset(payload: bytes):
    """Validate uploaded labeled BBC CSV; reject unlabeled competition test data.

    Canonical duplicate texts are removed BEFORE splitting. Conflicting labels
    fail loudly. English is a source property plus a heuristic screen, not a
    proven language detector. CSV bytes and audit counts are retained as metadata.
    """
    df = pd.read_csv(io.BytesIO(payload))
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={'content': 'text', 'articleid': 'article_id'})
    if not {'text', 'category'}.issubset(df.columns):
        raise ValueError('Upload BBC News Train.csv with Text and Category columns; Test.csv has no labels.')
    audit = {'sha256': hashlib.sha256(payload).hexdigest(), 'uploaded_rows': len(df)}
    df = df.dropna(subset=['text', 'category']).copy()
    audit['missing_removed'] = audit['uploaded_rows'] - len(df)
    df['text'] = df['text'].map(clean_text)
    df['category'] = df['category'].astype(str).str.strip().str.lower().replace({'sports': 'sport', 'technology': 'tech'})
    unknown = set(df.category) - set(CATEGORIES)
    if unknown:
        raise ValueError(f'Unexpected BBC labels: {sorted(unknown)}')
    df['canonical'] = df.text.str.casefold().str.replace(r'[^\w]+', ' ', regex=True).str.strip()
    if (df.groupby('canonical').category.nunique() > 1).any():
        raise ValueError('Duplicate text has conflicting labels; resolve the source before training.')
    before = len(df)
    df = df.drop_duplicates('canonical')
    audit['duplicates_removed'] = before - len(df)
    substantial = df.text.str.split().str.len().ge(40)
    # Detect grossly incompatible inputs; source provenance supplies language assurance.
    english_screen = df.text.str.lower().str.contains(r'\b(?:the|and|of|to|in|was|is|for)\b', regex=True)
    audit['short_or_language_screen_removed'] = int((~(substantial & english_screen)).sum())
    df = df[substantial & english_screen].copy()
    if df.category.nunique() < 4 or len(df) < 500:
        raise ValueError('Need at least 500 substantial English articles across four BBC categories.')
    if len(df) > 2000:
        df, _ = train_test_split(df, train_size=2000, stratify=df.category, random_state=SEED)
    if df.category.value_counts().min() < 10:
        raise ValueError('Each category needs at least ten valid articles for stratified evaluation.')
    df = df.reset_index(drop=True)
    df['row_id'] = np.arange(len(df))
    audit['retained_rows'] = len(df)
    audit['category_counts'] = df.category.value_counts().sort_index().to_dict()
    audit['uppercase_letter_fraction'] = sum(c.isupper() for t in df.text for c in t) / max(1, sum(c.isalpha() for t in df.text for c in t))
    return df, audit


def doc_features(doc, sia):
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    n = max(1, len(tokens))
    lemmas = [t.lemma_.lower() for t in tokens if t.is_alpha and not t.is_stop and len(t.lemma_) > 1]
    pos = Counter(t.pos_ for t in tokens)
    emotion = {k: sum(t.lemma_.lower() in words for t in tokens) for k, words in EMOTIONS.items()}
    dominant = max(emotion, key=emotion.get) if max(emotion.values()) else 'unmarked'
    compound = sia.polarity_scores(doc.text)['compound']
    syntax = {
        'sentence_length': len(tokens) / max(1, len(list(doc.sents))),
        'dependency_distance': np.mean([abs(t.i - t.head.i) for t in tokens]) if tokens else 0,
        'passive_rate': sum(t.dep_ in {'auxpass', 'nsubjpass'} for t in tokens) / n,
        'clause_rate': sum(t.dep_ in {'ccomp', 'xcomp', 'advcl', 'relcl'} for t in tokens) / n,
    }
    relations = []
    for verb in doc:
        if verb.pos_ != 'VERB':
            continue
        subjects = [t for t in verb.children if t.dep_ in {'nsubj', 'nsubjpass'}]
        objects = [t for t in verb.children if t.dep_ in {'dobj', 'obj', 'attr'}]
        agents = [c for t in verb.children if t.dep_ == 'agent' for c in t.children if c.dep_ == 'pobj']
        passive = any(t.dep_ == 'nsubjpass' for t in subjects)
        for subject in subjects:
            relations.append({'agent': ' '.join(t.text for t in agents) if passive else subject.text,
                              'action': verb.lemma_, 'patient': subject.text if passive else ' '.join(t.text for t in objects),
                              'negated': any(t.dep_ == 'neg' for t in verb.children), 'passive': passive})
    entities = [{'text': e.text, 'label': e.label_, 'start': e.start_char, 'end': e.end_char} for e in doc.ents if e.label_ in ENTITY_LABELS]
    pairs = Counter()
    for sentence in doc.sents:
        names = sorted({e.text.casefold() for e in sentence.ents if e.label_ in ENTITY_LABELS})
        pairs.update(combinations(names[:20], 2))
    return {'processed': ' '.join(lemmas), 'pos': {k: v/n for k,v in pos.items()},
            'patterns': dict(Counter(f'{a.pos_}-{b.pos_}' for a,b in zip(tokens, tokens[1:])).most_common(8)),
            'syntax': syntax, 'sentiment': compound,
            'sentiment_label': 'positive' if compound >= .05 else 'negative' if compound <= -.05 else 'neutral',
            'emotions': emotion, 'emotion': dominant, 'entities': entities, 'relations': relations[:20],
            'pairs': [{'source': a, 'target': b, 'count': int(c)} for (a,b),c in pairs.items()]}


def enrich(df, nlp, sia):
    """Parse in batches; dataset is capped at 2,000, documents at 100,000 chars."""
    if df.text.str.len().max() > 100_000:
        raise ValueError('Article exceeds the 100,000-character resource limit.')
    features = [doc_features(doc, sia) for doc in nlp.pipe(df.text, batch_size=24)]
    result = df.copy()
    for key in features[0]:
        result[key] = [f[key] for f in features]
    return result


class SyntaxFeatures(BaseEstimator, TransformerMixin):
    """Nonnegative, per-document grammar features; scaling is fitted per fold."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[*row['syntax'].values(), *[row['pos'].get(p,0) for p in ['NOUN','VERB','ADJ','PROPN','NUM']]] for row in X])


class TextField(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [row['processed'] for row in X]


def text_pipeline(model):
    return Pipeline([('text', TextField()), ('tfidf', TfidfVectorizer(max_features=10000, min_df=2, max_df=.95, sublinear_tf=True)),
                     ('select', SelectPercentile(chi2, percentile=100)), ('model', model)])


def hybrid_pipeline():
    return Pipeline([('features', FeatureUnion([
        ('lexical', Pipeline([('text', TextField()), ('tfidf', TfidfVectorizer(max_features=10000, min_df=2, max_df=.95, sublinear_tf=True))])),
        ('grammar', Pipeline([('extract', SyntaxFeatures()), ('scale', StandardScaler(with_mean=False))]))])),
        ('model', LogisticRegression(max_iter=1500, C=2, random_state=SEED))])


def make_splits(df):
    train, rest = train_test_split(np.arange(len(df)), test_size=.30, stratify=df.category, random_state=SEED)
    val, test = train_test_split(rest, test_size=.5, stratify=df.iloc[rest].category, random_state=SEED)
    assert not (set(train)&set(test) or set(train)&set(val) or set(val)&set(test))
    return train, val, test


def fit_models(df, splits):
    """Tune on train CV only. Validation tunes review threshold; test is untouched."""
    tr, va, te = splits
    records = df.to_dict('records')
    X = [records[i] for i in tr]
    y = df.iloc[tr].category.to_numpy()
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    models = {
        'Naive Bayes': (text_pipeline(MultinomialNB()), {'model__alpha':[.5,1.0]}),
        'Logistic regression': (text_pipeline(LogisticRegression(max_iter=1500, random_state=SEED)), {'model__C':[1,2]}),
        'Linear SVM': (text_pipeline(LinearSVC(random_state=SEED)), {'model__C':[.5,1]}),
    }
    fitted, rows = {}, []
    for name,(pipe,params) in models.items():
        params.update({'tfidf__ngram_range':[(1,1),(1,2)], 'select__percentile':[75,100]})
        start = time.perf_counter()
        search = GridSearchCV(pipe, params, scoring='f1_macro', cv=cv, n_jobs=1, error_score='raise')
        search.fit(X,y)
        fitted[name] = search.best_estimator_
        rows.append({'model':name,'cv_macro_f1':search.best_score_, 'cv_std':search.cv_results_['std_test_score'][search.best_index_],
                     'seconds':time.perf_counter()-start, 'parameters':str(search.best_params_)})
        print(f'{name}: CV macro-F1 {search.best_score_:.4f}', flush=True)
    table = pd.DataFrame(rows).sort_values('cv_macro_f1', ascending=False)
    selected = table.iloc[0]['model']
    return fitted, table, selected


def score_models(df, splits, fitted):
    te = splits[2]
    X = df.iloc[te].to_dict('records')
    y = df.iloc[te].category.to_numpy()
    rows, reports, predictions = [], {}, {}
    for name, model in fitted.items():
        pred = model.predict(X)
        rows.append({'model':name,'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'), 'weighted_f1':f1_score(y,pred,average='weighted')})
        reports[name] = classification_report(y,pred,output_dict=True,zero_division=0)
        predictions[name] = pred
    return pd.DataFrame(rows), reports, predictions


def paired_bootstrap(y, a, b, repeats=500):
    rng = np.random.default_rng(SEED)
    classes = np.unique(y)
    differences = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(np.flatnonzero(y==c), size=sum(y==c), replace=True) for c in classes])
        differences.append(f1_score(y[idx], b[idx], average='macro') - f1_score(y[idx], a[idx], average='macro'))
    return [float(x) for x in np.quantile(differences,[.025,.975])]


def research_experiment(df, splits):
    """Predeclared ablation: identical LR, unigram TF-IDF, with/without grammar."""
    tr,va,te = splits
    base = text_pipeline(LogisticRegression(C=2,max_iter=1500,random_state=SEED))
    hybrid = hybrid_pipeline()
    Xtr = df.iloc[tr].to_dict('records')
    ytr = df.iloc[tr].category.to_numpy()
    Xte = df.iloc[te].to_dict('records')
    yte = df.iloc[te].category.to_numpy()
    base.fit(Xtr,ytr)
    hybrid.fit(Xtr,ytr)
    a,b = base.predict(Xte), hybrid.predict(Xte)
    # A label-preserving stress test, not a second natural test set.
    masked = []
    for row in Xte:
        copy = dict(row)
        words = {w.lower() for e in row['entities'] for w in e['text'].split()}
        copy['processed'] = ' '.join(w for w in row['processed'].split() if w not in words)
        masked.append(copy)
    return {'hypothesis':'Adding grammar improves macro-F1 beyond lexical features.',
            'baseline_f1':f1_score(yte,a,average='macro'), 'hybrid_f1':f1_score(yte,b,average='macro'),
            'difference_ci95':paired_bootstrap(yte,a,b),
            'baseline_masked_f1':f1_score(yte,base.predict(masked),average='macro'),
            'hybrid_masked_f1':f1_score(yte,hybrid.predict(masked),average='macro'),
            'caveat':'Single split; bootstrap conditions on fitted models. Masked grammar retains original syntax. No claim of academic novelty or causal benefit.'}


def review_threshold(df, splits, model):
    """Validation-only margin threshold; margin is not a calibrated probability."""
    X = df.iloc[splits[1]].to_dict('records')
    if hasattr(model,'decision_function'):
        scores = model.decision_function(X)
    else:
        scores = model.predict_proba(X)
    ordered = np.sort(scores,axis=1)
    margins = ordered[:,-1]-ordered[:,-2]
    threshold = float(np.quantile(margins,.20))
    return threshold


def analyze_article(text, model, nlp, sia, threshold, custom_ner=None):
    text = clean_text(text)
    if len(text.split()) < 8:
        raise ValueError('Please provide at least eight words of article text.')
    if len(text) > 100_000:
        raise ValueError('Please keep articles below 100,000 characters.')
    row = doc_features(nlp(text),sia)
    if not row['processed']:
        raise ValueError('No usable English content words were found.')
    category = str(model.predict([row])[0])
    scores = model.decision_function([row]) if hasattr(model,'decision_function') else model.predict_proba([row])
    ordered = np.sort(scores[0])
    margin = float(ordered[-1]-ordered[-2])
    row.update({'predicted_category':category,'score_margin':margin,'review_required':bool(margin<threshold),
                'scope':'Five BBC labels; health and non-English input are outside the trained label space.'})
    row['custom_entities'] = [{'text':e.text,'label':e.label_} for e in custom_ner(text).ents] if custom_ner else []
    return row


def create_dashboard(model,nlp,sia,threshold,custom_ner=None):
    import gradio as gr
    def predict(text):
        try:
            result = analyze_article(text,model,nlp,sia,threshold,custom_ner)
            return (f"{result['predicted_category'].title()} | sentiment: {result['sentiment_label']} | "
                    f"review: {result['review_required']}", result)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
    return gr.Interface(fn=predict, inputs=gr.Textbox(lines=8,label='English news article'),
                        outputs=[gr.Textbox(label='News desk summary'),gr.JSON(label='Entities, syntax, tone and review evidence')],
                        title='NewsBot Intelligence Desk',
                        description='Five BBC categories. VADER tone and emotion cues are heuristic; verify facts and entity links.',
                        examples=[["A football club won the league final after its striker scored two goals. The coach praised the team and supporters celebrated the victory at the stadium."],
                                  ["A software company unveiled a new processor for laptops. Engineers said the chip improves computing speed and battery efficiency. The product will reach computer manufacturers next month."]])
