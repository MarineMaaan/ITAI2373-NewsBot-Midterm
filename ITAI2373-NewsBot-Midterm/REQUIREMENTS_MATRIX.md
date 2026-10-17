# Assignment traceability

| Requirement | Implementation | Evidence |
|---|---|---|
| Module 1: application context | Notebook §1: assisted editorial routing; studio review adaptation | Scope, target operator, value and limits in notebook/report |
| Module 2: preprocessing | `clean_text`, `enrich`, `doc_features`; missing/duplicate/language/length checks | Audit, token reduction, split manifest; focused edge-case tests |
| Module 3: TF-IDF | Pipeline CV: n-grams, feature selection, model settings | Category-term charts, model-feature weights, CV table, sparsity |
| Module 4: POS | Mean per-article proportions and POS bigrams | POS category heatmap and tables |
| Module 5: syntax/semantics | Dependency features; heuristic agent/action/patient, passive voice, negation | Syntax tables/chart and explicit relation demonstration |
| Module 6: sentiment/emotion | VADER article sentiment; four emotion cue lexicons | Per-category distributions; limitations documented |
| Module 7: multiclass classification | NB, logistic regression, linear SVM; three-fold CV | Reserved-test metrics, per-class report, confusion matrix, error table |
| Module 8: NER | PERSON, ORG, GPE, DATE, MONEY | Entity frequencies/category heatmap/co-occurrence network |
| Data constraints | Explicit CSV upload; 500–2,000 substantial English-source articles, ≥4 categories | Source audit and validation; actual BBC five classes |
| Cohesive pipeline and new input | `analyze_article` reused by notebook and dashboard | Two original new-article demonstrations |
| Bonus: dashboard (5) | Gradio textbox, summary and structured output | Constructed in notebook; browser and API validation recorded separately |
| Bonus: advanced analysis (5) | 25 English/Spanish paired probes; fixed glossary aid | Paired predictions, accuracy/macro-F1 and chart |
| Bonus: custom model (5) | Trained spaCy NEWS_OUTLET/TECH_PRODUCT NER | Disjoint train/dev/test names and frames; learning curve, exact-span P/R/F1, test predictions |
| Bonus: research (10) | Project-specific lexical+grammar hybrid hypothesis, ablation and entity-masking stress | Matched baseline, macro-F1, 500-resample paired interval; interpretation |
| Public individual portfolio repository | MarineMaaan repository; named project subfolder | Repository URL in README, notebook, journal and technical report |
| Specific contributions and citations | Solo contribution/AI assistance statement; linked primary sources | README and notebook sources; student's reflection themes |
| Reflective journal | Editable Word journal plus PDF | Exactly two rendered pages; all six reflection topics |
| Written report and naming | `MT_Report_Benjamin_LaCount_II_ITAI2373` | DOCX/PDF with working GitHub hyperlink; group token omitted for solo work |
| No video or group identity | Student-directed adaptations | No fabricated group or video; four non-video bonus categories retained |

The assignment says up to 15 bonus points while listing 25 points across four categories. All four are implemented; the instructor determines eligible points. The optional video bonus conflicts with the student's explicit no-video requirement and is excluded.

The journal addresses team collaboration as independent role coordination, technical integration challenges, business value, individual contributions, future enhancements and professional development. The student's supplied notes ground the personal statements.
