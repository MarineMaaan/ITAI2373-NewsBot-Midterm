"""Bonus experiments. Authored evaluation sets are controlled demonstrations.

They do not establish real-world multilingual accuracy or production NER quality.
"""
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, f1_score
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

from newsbot import SEED, doc_features

# Fixed before evaluation. These pairs are original teaching examples, not BBC text.
# The glossary is an experimental rule-based translation aid, not a translator.
GLOSSARY = {
    'empresa':'company','acciones':'shares','mercado':'market','beneficios':'profits','banco':'bank',
    'inversores':'investors','economia':'economy','ventas':'sales','crecimiento':'growth','precio':'price',
    'equipo':'team','partido':'match','gol':'goal','futbol':'football','liga':'league','jugador':'player',
    'entrenador':'coach','estadio':'stadium','victoria':'victory','torneo':'tournament',
    'gobierno':'government','elecciones':'election','ministro':'minister','parlamento':'parliament','votantes':'voters',
    'partido_politico':'party','ley':'law','politica':'policy','presidente':'president','votos':'votes',
    'pelicula':'film','actor':'actor','musica':'music','cine':'cinema','premio':'award',
    'cantante':'singer','album':'album','concierto':'concert','actriz':'actress','festival':'festival',
    'software':'software','ordenador':'computer','internet':'internet','tecnologia':'technology','usuarios':'users',
    'seguridad':'security','procesador':'processor','datos':'data','red':'network','movil':'mobile',
}
PAIR_SPECS = {
    'business': [('empresa acciones mercado beneficios','company shares market profits'),
                 ('banco inversores economia crecimiento','bank investors economy growth'),
                 ('ventas beneficios empresa precio','sales profits company price'),
                 ('mercado acciones inversores banco','market shares investors bank'),
                 ('economia crecimiento ventas empresa','economy growth sales company')],
    'sport': [('equipo partido gol futbol','team match goal football'),('liga jugador entrenador victoria','league player coach victory'),
              ('estadio futbol equipo torneo','stadium football team tournament'),('jugador gol liga partido','player goal league match'),
              ('entrenador equipo victoria estadio','coach team victory stadium')],
    'politics': [('gobierno elecciones ministro parlamento','government election minister parliament'),
                 ('votantes partido_politico politica votos','voters party policy votes'),
                 ('presidente gobierno ley parlamento','president government law parliament'),
                 ('elecciones votos ministro politica','election votes minister policy'),
                 ('parlamento votantes gobierno presidente','parliament voters government president')],
    'entertainment': [('pelicula actor cine premio','film actor cinema award'),('musica cantante album concierto','music singer album concert'),
                      ('actriz pelicula festival premio','actress film festival award'),('cine actor pelicula festival','cinema actor film festival'),
                      ('album musica cantante premio','album music singer award')],
    'tech': [('software ordenador internet tecnologia','software computer internet technology'),
             ('usuarios seguridad datos red','users security data network'),('procesador ordenador tecnologia software','processor computer technology software'),
             ('movil internet usuarios red','mobile internet users network'),('datos seguridad software tecnologia','data security software technology')],
}


def cross_lingual(model,nlp,sia):
    import unicodedata
    rows = []
    def predict(t):
        return model.predict([doc_features(nlp(t),sia)])[0]
    for label,pairs in PAIR_SPECS.items():
        for i,(es,en) in enumerate(pairs):
            spanish = f'El informe analiza {es.replace(" ",", ")} en el sector. Los expertos presentan los resultados de una nueva investigacion.'
            english = f'The report examines {en.replace(" ",", ")} in the sector. Experts present the results of a new investigation.'
            normalized = ''.join(c for c in unicodedata.normalize('NFD',spanish.lower()) if not unicodedata.combining(c))
            translated = re.sub(r'\b\w+\b',lambda m:GLOSSARY.get(m.group(),m.group()),normalized)
            rows.append({'pair_id':f'{label}-{i}','expected':label,'english':english,'spanish':spanish,
                         'english_prediction':predict(english),'spanish_prediction':predict(spanish),'glossary_prediction':predict(translated)})
    table = pd.DataFrame(rows)
    metrics = []
    for condition,col in [('English control','english_prediction'),('Spanish direct','spanish_prediction'),('Spanish with glossary','glossary_prediction')]:
        metrics.append({'condition':condition,'accuracy':accuracy_score(table.expected,table[col]),'macro_f1':f1_score(table.expected,table[col],average='macro')})
    return table,pd.DataFrame(metrics)


def ner_annotations():
    """Names and sentence frames are disjoint across train/dev/test.

    Two custom newsroom labels; 96 train, 12 dev, 12 test sentences. Offsets are
    built from exact strings and checked against tokenizer boundaries. Entire
    outlet/product pairs are held out, avoiding a memorized-name-only evaluation.
    """
    names = {
        'train': [('Pixel Ledger','Aurora Engine'),('Studio Signal','Prism Console'),('Game Wire','Vertex Toolkit'),
                  ('Render Review','Nova Headset'),('Play Chronicle','Orbit Controller'),('Indie Beacon','Vivid Editor'),
                  ('Quest Tribune','Pulse Handheld'),('Digital Gazette','Helix Runtime')],
        'dev': [('Arcade Observer','Cobalt Platform'),('Frame Bulletin','Ember Compiler')],
        'test': [('Level Dispatch','Nimbus Device'),('Creator Sentinel','Quartz Renderer')],
    }
    templates = {
        'train': ['{o} reviewed {p} for game developers.', '{o} published a report about {p}.',
                  'Developers told {o} that {p} supports their studio.', '{p} received an update according to {o}.',
                  'An editor at {o} tested {p} yesterday.', 'The launch of {p} was covered by {o}.',
                  '{o} compared the performance of {p}.', 'A journalist from {o} discussed {p} at the conference.',
                  '{p} appeared in a feature from {o}.', '{o} interviewed the engineers behind {p}.',
                  'The studio demonstrated {p} to {o}.', '{o} reported that {p} will ship next month.'],
        'dev': ['A new investigation by {o} examined {p}.','During the briefing, {o} asked about {p}.',
                'Readers of {o} requested a test of {p}.','{p} attracted attention from {o} this week.',
                'The team invited {o} to inspect {p}.','{o} offered an assessment of {p}.'],
        'test': ['In an exclusive story, {o} described the release of {p}.','The makers of {p} spoke with {o} after the event.',
                 'Coverage from {o} highlighted changes to {p}.','Following a demonstration of {p}, {o} published its findings.',
                 '{o} announced plans to investigate the design of {p}.','A detailed evaluation of {p} appeared in {o} on Monday.'],
    }
    rows=[]
    for split,pairs in names.items():
        for outlet,product in pairs:
            for template in templates[split]:
                text=template.format(o=outlet,p=product)
                entities=[]
                for entity,label in [(outlet,'NEWS_OUTLET'),(product,'TECH_PRODUCT')]:
                    start=text.index(entity)
                    entities.append((start,start+len(entity),label))
                rows.append({'split':split,'text':text,'entities':sorted(entities)})
    return rows


def train_custom_ner(output_dir='artifacts/custom_ner'):
    fix_random_seed(SEED)
    rng=random.Random(SEED)
    rows=ner_annotations()
    model=spacy.blank('en')
    ner=model.add_pipe('ner')
    for label in ['NEWS_OUTLET','TECH_PRODUCT']:
        ner.add_label(label)
    def examples(split):
        result=[]
        for row in rows:
            if row['split'] != split:
                continue
            doc=model.make_doc(row['text'])
            for a,b,_ in row['entities']:
                assert doc.char_span(a,b,alignment_mode='strict') is not None
            result.append(Example.from_dict(doc,{'entities':row['entities']}))
        return result
    train=examples('train')
    optimizer=model.initialize(lambda:train)
    best=-1
    checkpoint=None
    history=[]
    for epoch in range(25):
        rng.shuffle(train)
        losses={}
        for batch in minibatch(train,size=8):
            model.update(batch,sgd=optimizer,drop=.2,losses=losses)
        metrics=model.evaluate(examples('dev'))
        history.append({'epoch':epoch+1,'loss':losses.get('ner',0),'dev_f1':metrics['ents_f']})
        if metrics['ents_f']>best:
            best=metrics['ents_f']
            checkpoint=model.to_bytes()
    model.from_bytes(checkpoint)
    test=model.evaluate(examples('test'))
    # Untrained domain-label baseline: emits no custom entities.
    summary={'train_sentences':96,'dev_sentences':12,'test_sentences':12,
             'test_precision':test['ents_p'],'test_recall':test['ents_r'],'test_f1':test['ents_f'],
             'per_label':test['ents_per_type'],'no_custom_entity_baseline_f1':0.0,
             'selected_epoch':next(r['epoch'] for r in history if r['dev_f1']==best),
             'limitation':'Authored synthetic sentences, two unseen name pairs and six unseen frames; no manually annotated BBC NER test set.'}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.to_disk(output_dir)
    predictions=[]
    for row in rows:
        if row['split']=='test':
            pred=model(row['text'])
            predictions.append({'text':row['text'],'gold':row['entities'],'predicted':[(e.start_char,e.end_char,e.label_) for e in pred.ents]})
    return model,summary,pd.DataFrame(history),rows,predictions
