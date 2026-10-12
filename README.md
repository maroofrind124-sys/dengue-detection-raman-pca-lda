# Dengue Detection from Raman Spectra using PCA + LDA

Machine learning pipeline for detecting dengue fever from Raman spectra of serum samples, built as a Final Year Project (BS Computer Science).

The pipeline uses **StandardScaler → PCA → SMOTE → Linear Discriminant Analysis (LDA)** and is evaluated with **patient-wise nested cross-validation**, so no patient ever appears in both the training and test data.



---

## Results

Evaluated on all 38 patients using nested, patient-wise cross-validation (5 repeated runs with different fold splits, mean ± std):

| Metric | Spectrum-level | 
|---|---|
| Accuracy | 96.37% ± 0.69 |
| Balanced Accuracy | 96.24% ± 0.68 |
| Sensitivity | 97.59% ± 1.15 |
| Specificity | 94.89% ± 1.08 |
| AUC | 0.99 |

**Patient-level accuracy: 97.37% (37 / 38 patients correct)**, obtained by averaging the predicted probability over all spectra of a patient. Only one healthy patient is misclassified.

Reference result from the literature (PCA-LDA on Raman spectra of dengue sera): Accuracy 96.50%, Sensitivity 93.44%, Specificity 100%. Our accuracy is comparable and our sensitivity is higher, while specificity is lower. Note that the datasets and validation schemes differ, so this is not a direct head-to-head comparison.

> Exact numbers can vary by about ±1% depending on library versions and how folds are assigned.

---

## Dataset

- 760 Raman spectra from **38 patients** (23 dengue, 15 healthy), 20 spectra per patient
- 666 spectral features per spectrum (`wn_<wavenumber>` columns, approx. 546–1827 cm⁻¹)
- Each spectrum is L2-normalized (done per spectrum, so it does not leak information across samples)
- 138 duplicate spectra (all inside the same patient) were removed, leaving **622 unique spectra** (340 dengue, 282 healthy)

Expected CSV format: columns `patient_id`, `label` (1 = dengue, 0 = healthy), and the `wn_*` feature columns.

---

## Methodology

1. **Preprocessing:** fill missing values (none found), remove duplicate spectra
2. **Patient-wise cross-validation:** `StratifiedGroupKFold` (8 outer folds, 4 inner folds) grouped by `patient_id`
3. **Pipeline (fit on training folds only):**
   - `StandardScaler`
   - `PCA`, with `n_components` chosen from `[5, 10, 20, 30, 50, 80, 100]` by inner cross-validation (balanced accuracy)
   - `SMOTE` for class balancing (applied to training data only, never to test data)
   - `LinearDiscriminantAnalysis`
4. **Evaluation:** nested CV repeated with 5 random seeds, plus patient-level aggregation
5. **Final model:** retrained on all data and saved as `dengue_model.pkl` (scaler + PCA + LDA)

### Why these choices

- **Patient-wise splits** prevent leakage between spectra of the same patient. An earlier version without this gave an inflated accuracy of ~99.7%, which is why honest validation was used.
- **Nested CV** keeps the choice of `n_components` from biasing the final score.
- After duplicate removal the classes are close to balanced (about 1.2 : 1), so SMOTE is used as a safeguard and has only a small effect.

### Sanity checks

- **Label permutation test:** shuffling patient labels drops accuracy to chance level (about 39% to 64% across 4 runs), which indicates the pipeline does not leak information.
- **Cross-patient duplicates:** no exact or near-exact copies of spectra between different patients.

---

## Repository structure

```
.
├── dengue_PCA_LDA_final_dataset.ipynb   # full step-by-step notebook
├── dengue_model.pkl                     # trained model (scaler + PCA + LDA)
├── requirements.txt
└── README.md
```

---

## How to run

### Google Colab (recommended)
1. Open `dengue_PCA_LDA_final_dataset.ipynb` in Colab
2. Set `DATA_FILE = 'dengue_preprocessed.csv'` in the first code cell
3. Run all cells and upload the CSV when asked

### Locally
```bash
pip install -r requirements.txt
jupyter notebook dengue_PCA_LDA_final_dataset.ipynb
```

---

## Using the saved model

```python
import pickle
import pandas as pd

pkg = pickle.load(open('dengue_model.pkl', 'rb'))   # only load files you trust

new = pd.read_csv('new_spectra.csv')                # must contain the same wn_* columns
X = new[pkg['wn_cols']].values

X = pkg['pca'].transform(pkg['scaler'].transform(X))
prob = pkg['lda'].predict_proba(X)[:, 1]            # probability of dengue
print(prob)
```

The model was saved with the scikit-learn version listed in `requirements.txt`; use a similar version to avoid loading problems. Spectra must be preprocessed the same way as the training data (L2 normalization).

---

## Limitations

- Small dataset (38 patients), so results can change with a few patients
- One healthy patient is consistently misclassified (an outlier)
- Internal validation only; the model has not been tested on data from another instrument, protocol, or hospital

---

## Tech stack

Python, NumPy, pandas, scikit-learn, imbalanced-learn, Matplotlib, Seaborn

## Author

**Maroof**, BS Computer Science (Final Year)
[LinkedIn / GitHub / Email: add your links here]
