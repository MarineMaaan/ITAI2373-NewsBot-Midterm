# Data provenance and handling

- Source: [BBC News Classification competition](https://www.kaggle.com/competitions/learn-ai-bbc/data).
- Required input: the labeled **BBC News Train.csv**, with `ArticleId`, `Text`, `Category` columns. Capitalization of column names is normalized.
- Student confirmed instructor approval for this dataset on October 17, 2026.
- Download the archive yourself from Kaggle, accept the site's applicable conditions, extract it and upload the training CSV in the notebook cell calling `google.colab.files.upload()`.
- The competition test CSV is unlabeled. The sample solution is a format example, not ground truth. Neither is used to train or evaluate.
- The five source categories are business, entertainment, politics, sport and tech. Health is unsupported.
- Files remain in the user's runtime; no training text, credentials or Kaggle token is committed. Colab uploads disappear when the runtime resets. The dashboard's temporary share URL expires with its runtime.
- The notebook drops missing text/labels, canonical exact duplicates and articles shorter than 40 words. Conflicting duplicate labels fail validation. A lightweight English screen supplements source provenance; it is not language identification.
- A fixed seed selects up to 2,000 articles; at least 500 and four categories are required. The actual five-label source distribution, source-byte SHA-256 and retained row count are recorded in `outputs/results.json`.
- Linguistic analyses use the retained corpus descriptively. Models learn vocabulary and selected features only from training folds. Test labels are reserved for evaluation. No publication dates or full titles are invented.
- Authored bilingual probes and custom NER annotations are separate instructional assets; they are explicitly identified as synthetic and are not BBC examples.
- Original news content and dataset rights remain with their owners. Repository publication does not grant redistribution rights to the dataset.
