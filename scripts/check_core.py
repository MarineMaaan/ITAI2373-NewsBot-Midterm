"""Focused behavior checks for leakage boundaries, bad uploads and role extraction."""
from pathlib import Path
import io
import sys
import unittest
import numpy as np
import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ITAI2373-NewsBot-Midterm'))
from newsbot import read_dataset, clean_text, make_splits, text_pipeline, doc_features, analyze_article
from sklearn.linear_model import LogisticRegression
from extensions import ner_annotations

class BehaviorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp=spacy.load('en_core_web_sm'); cls.sia=SentimentIntensityAnalyzer()

    def test_clean_edge_cases(self):
        self.assertEqual(clean_text(None),'')
        self.assertEqual(clean_text(np.nan),'')
        self.assertEqual(clean_text('<p>A &amp; B</p> https://example.org'),'A & B')
        self.assertIn('$20',clean_text('Studio earned $20 on Monday.'))

    def test_unlabeled_csv_rejected(self):
        with self.assertRaisesRegex(ValueError,'Train.csv'):
            read_dataset(b'ArticleId,Text\n1,Some article\n')

    def test_small_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError,'500'):
            read_dataset(b'Text,Category\nshort,business\n')

    def test_conflicting_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError,'conflicting'):
            read_dataset(b'Text,Category\nThe same text.,business\nTHE SAME TEXT!,sport\n')

    def test_nonoverlapping_stratified_splits(self):
        # Synthetic indices test partition behavior without evaluating a model.
        df=pd.DataFrame({'category':np.repeat(['a','b','c','d','e'],100)})
        splits=make_splits(df)
        self.assertEqual(sorted(np.concatenate(splits)),list(range(500)))
        self.assertEqual([len(i) for i in splits],[350,75,75])
        for idx in splits: self.assertEqual(df.iloc[idx].category.nunique(),5)

    def test_vectorizer_never_sees_test_only_word(self):
        pipe=text_pipeline(LogisticRegression())
        X=[{'processed':'bank profit company'},{'processed':'bank market company'},
           {'processed':'football goal team'},{'processed':'football player team'}]
        pipe.fit(X,['business','business','sport','sport'])
        before=set(pipe.named_steps['tfidf'].vocabulary_)
        pipe.predict([{'processed':'unseenholdouttoken football'}])
        self.assertEqual(before,set(pipe.named_steps['tfidf'].vocabulary_))
        self.assertNotIn('unseenholdouttoken',before)

    def test_passive_roles_and_negation(self):
        result=doc_features(self.nlp('The game was reviewed by the editor. The developer did not cancel the launch.'),self.sia)
        passive=next(r for r in result['relations'] if r['passive'])
        self.assertEqual(passive['patient'],'game'); self.assertEqual(passive['agent'],'editor')
        self.assertTrue(any(r['negated'] for r in result['relations']))

    def test_custom_ner_partition_disjointness(self):
        rows=ner_annotations()
        names={}
        sentences={}
        for split in ['train','dev','test']:
            selected=[r for r in rows if r['split']==split]
            names[split]={r['text'][a:b] for r in selected for a,b,label in r['entities']}
            sentences[split]={r['text'] for r in selected}
            for r in selected:
                for a,b,_ in r['entities']:
                    self.assertIsNotNone(self.nlp.make_doc(r['text']).char_span(a,b))
        for a,b in [('train','test'),('train','dev'),('dev','test')]:
            self.assertFalse(names[a]&names[b]); self.assertFalse(sentences[a]&sentences[b])

    def test_empty_article_validation(self):
        with self.assertRaisesRegex(ValueError,'eight words'):
            analyze_article('',None,self.nlp,self.sia,0)

if __name__=='__main__': unittest.main(verbosity=2)
