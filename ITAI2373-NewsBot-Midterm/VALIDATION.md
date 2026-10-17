# Validation record · October 17, 2026

## Executed evidence

- The final 34-cell notebook ran from start to finish in a **fresh local Python 3.11.9 CPU kernel**: all **18 code cells** executed without errors. Analysis runtime was **211.7 seconds**, excluding dependency installation. The first attempt revealed a missing model-save parent directory; that defect was fixed before this complete fresh run.
- Local execution used an explicitly supplied BBC CSV path. No automatic data download or substitute sample dataset was used. The local dataset SHA-256 is `f56a65dab2ea728505a639d2186a2b2f1c0939f28e8e89b5ed1c1fe9520b1323`.
- Input: **1,490** labeled training rows. Removed **52** canonical duplicates; retained **1,438** substantial articles with no missing text/category values across five categories. Split: **1,006 training / 216 validation / 216 test**. The unlabeled test file and sample solution were unused.
- Three-fold training CV selected **linear SVM**. Test accuracy **0.981481**, macro-F1 **0.980763**. Four held-out articles were misclassified. Other algorithms, per-category precision/recall/F1, confusion and errors are saved.
- Review threshold was selected from validation margins. On test data it accepted **75%** of articles, with accuracy **1.000** on that accepted subset. This selective result is conditional on this split; it is not an accuracy guarantee or an out-of-domain safety check.
- Custom NER used **96 train / 12 development / 12 test** synthetic sentences, with disjoint name pairs and sentence frames. The best development checkpoint was epoch 5. Test exact-span precision/recall/F1: **0.625 / 0.625 / 0.625**. Per-label results and all test predictions are included. This does not establish real article NER accuracy.
- Cross-lingual comparison used **25** authored English/Spanish pairs: English accuracy **0.960**, direct Spanish **0.240**, fixed-glossary Spanish **0.880**. Vocabulary overlap between the probe and glossary is intentional, so this is an optimistic controlled mechanism test.
- The grammar-hybrid research experiment produced macro-F1 **0.929188**, below the matched lexical baseline **0.970608**. The 500-resample stratified paired 95% interval for hybrid minus baseline was **[-0.073991, -0.014092]**. The evidence suggests a decline on this split, so the main classifier remains the CV-selected lexical SVM. Entity-word masking also favored the lexical baseline.

## Behavioral checks

- `scripts/check_core.py`: **9 tests passed**, covering cleanup, malformed/unlabeled uploads, inadequate datasets, conflicting duplicates, split coverage/separation, vocabulary leakage, passive/negated relations, NER annotation/split boundaries, and empty article validation.
- `scripts/check_notebook.py`: notebook schema valid; all code cells executed; no error outputs; embedded implementation exactly matches the repository source. The actual upload-cell code was exercised with a controlled `google.colab.files.upload()` stub, including empty, multiple-file and wrong-extension errors.
- The actual Gradio interface was opened in the in-app browser. A new original sports article and a new original technology article were entered and submitted using visible controls. The page displayed **Sport** and **Tech**, respectively, with tone and structured analysis. Screenshots: [sports](outputs/dashboard_sport.png), [technology](outputs/dashboard_tech.png).
- `scripts/check_dashboard.py`: the running `/predict` endpoint returned the expected two categories and integrated result fields. Blank input returned the expected readable validation error. [API evidence](outputs/dashboard_api_validation.json).
- All **12 analysis figures** were visually inspected. The screenshots and original new-article examples contain authored text, not whole BBC articles.

## Documents

- Editable reflective journal: **2 pages**. Technical report: **3 pages**. Each has a clickable public repository hyperlink.
- Documents were authored with `python-docx`. This Windows runtime has no bundled LibreOffice; a dedicated hidden Microsoft Word instance exported PDFs, then all five PDF pages were rendered to PNG and visually checked. No user Word document was opened or modified.
- Both the Word pagination calculation and independent PDF page counts agree. No clipping, overlapping text, blank overflow pages or placeholders remain.
- The journal's personal themes came from the student: preprocessing quality, lengthy preparation and future efficiency, studio game-review monitoring, and AI-assisted execution with student thought/planning/final QA responsibility. No team interactions were invented.

## Evidence boundaries

- This is **local CPU and local browser evidence**. A signed-in Google Colab session was not executed. The Colab setup/upload branch is implemented and its upload handling was checked with a stub; that is not a live Colab acceptance claim.
- Sentiment, emotions, standard NER and heuristic semantic roles have no gold BBC annotations; no accuracy is claimed for them. Original BBC text is entirely lowercase, which limits case-sensitive NER.
- BBC is a historical single-source benchmark. Synthetic extension tests, a single classifier split, exact rather than near-duplicate removal, missing publication metadata and possible source-specific artifacts limit generalization.
- The four non-video extra-credit categories are implemented and evaluated. The instructor controls points, including the stated 15-point cap. No video was created.
- Dataset approval was confirmed by the student. No instructor communication or Canvas submission was performed.
- Raw CSVs, credentials, virtual environment, generated runtime/model artifacts, the untouched starter and document QA intermediates are excluded from Git. The runtime/model artifacts can be regenerated by executing the notebook.
