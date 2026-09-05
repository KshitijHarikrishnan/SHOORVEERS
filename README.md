# Data ML — Submission Template

## 1. Team Details

| Field | Details |
| --- | --- |
| Team Name | SHOORVEERS |
| Team Members (4) | 1. KSHITIJ HARIKRISHNAN  2. MALLOBHA SAI MOURYA  3. JAKKA LOKESH REDDY  4. DASARI KARTHIK REDDY |
| Team Lead (contact) | KSHITIJ HARIKRISHNAN , (PhNo.) - 9676249876 |
| GitHub Repository Link | https://github.com/KshitijHarikrishnan/SHOORVEERS |

## 2. Repository Structure

```
/code
  ├── 01_eda.ipynb
  ├── 02_preprocessing.ipynb
  ├── 03_feature_extraction.ipynb
  ├── 04_modelling.ipynb
  └── /Outputs
        ├── test_features.csv
        ├── train_features.csv
        ├── feature_importance.csv
        ├── calibrated_scores.csv
        ├── baseline_scores.csv
        ├── baseline_vs_blob_comparison.csv
        ├── validation_accuracy_vs_auroc.csv
        └── validation_metrics_percentage.csv
  └── /xray_images
        ├── Atelectasis_xray.png
        ├── Effusion_xray.png
        ├── Infiltration_xray.png
        ├── Nodule_xray.png
        └── Pneumothorax_xray.png
/predictions
  └── predictions.csv
/Wildcard Challenges
  └── WC_01
        ├── WC_01.ipynb
        └── WC_01.png
README.md
requirements.txt
.gitignore
```

 Raw train/test image folders are excluded from version control (documented in `.gitignore`) due to size. Derived feature tables (`train_features.csv`, `test_features.csv`) are committed so results are reproducible without re-running feature extraction from scratch.

## 3. Problem Understanding

The task is to analyze chest X-ray images and predict the occurence of five conditions: Atelectasis, Effusion, Infiltration, Nodule, and Pneumothorax. Since an X-ray can show multiple conditions at once, we are asked to generate an independent probability score (0–1) for each condition. Basically, this probability score answers the question, "does this image show Atelectasis: yes/no" for each condition. There is a constraint that rules out deep learning and pretrained models as the main challenge is to extract meaningful features (ex: texture changes, shape irregularities, left-right asymmetry etc) from the X-ray images, which can then be used with classical machine-learning models. The classic ML model, learns from the train data and predicts outputs for the test data.

## 4. Exploratory Data Analysis (EDA)

Our EDA (see `01_eda.ipynb`) clarly shows significant class imbalance across all five conditions, with each condition positive in roughly 5-15% of training images — Nodule and Pneumothorax were the rarest, Atelectasis and Infiltration relatively more common. This imbalance shaped every downstream decision: we used `class_weight='balanced'` during model training, avoided raw accuracy as a primary metric (a naive "always predict negative" model would score 90%+ "accuracy" on rare conditions while learning nothing), and ultimately tuned per-condition classification thresholds rather than using a fixed 0.5 cutoff.

We also found a small number of images (49 of 4,484 merged training records, and 34 of 1,122 test images) produced missing (NaN) feature values, traced to degenerate image regions — patches of near-uniform pixel intensity after lung segmentation where statistical measures like skewness and kurtosis are mathematically undefined. For the training set we dropped these rows (a small, acceptable loss); for the test set, where every image requires a prediction, we imputed using training-set medians to avoid leaking test-set information into our own preprocessing.

## 5. Approach & Methodology

### 5.1 Feature Engineering

All features are computed on lung regions isolated via Otsu thresholding and morphological cleanup (`02_preprocessing.ipynb`), so texture and shape features reflect lung tissue rather than borders, labels, or surrounding anatomy. We engineered four feature families, each targeted at how a specific condition visually manifests:

- **GLCM texture features** (contrast, homogeneity, energy) — capture tissue roughness/smoothness, relevant to diffuse density changes seen in Infiltration and Effusion.
- **HOG (histogram of oriented gradients) features** — capture edge and shape structure, relevant to the more sharply bounded opacities seen in Atelectasis and Nodule.
- **Regional intensity statistics** (mean, std, skew, kurtosis), computed separately for upper/middle/lower thirds of each lung — designed specifically for Effusion, since pleural fluid accumulates at the lung base under gravity.
- **Left-right symmetry score** — many conditions, particularly Pneumothorax, tend to be unilateral, so we directly compared left vs. mirrored-right lung statistics.
- **Blob detection features** (`num_blobs`, `max_blob_size`, `mean_blob_size` etc) — added specifically to target Nodule, which presents as small, round, well-defined opacities that our other feature families weren't designed to isolate.

### 5.2 Model(s) Used

We trained one binary classifier per condition (5 independent binary-relevance models), comparing three classical model types per condition: Logistic Regression, Random Forest, and XGBoost. We selected whichever model scored highest validation AUROC per condition — Logistic Regression won for Atelectasis and Effusion, Random Forest for Infiltration, Nodule, and Pneumothorax. All final models were probability-calibrated using isotonic regression (`CalibratedClassifierCV`) so output scores reflect genuine probabilities. We evaluated with AUROC as our primary metric given the class imbalance, and additionally tuned per-condition classification thresholds (rather than a fixed 0.5) by maximizing balanced accuracy on the validation set — this corrected a discovered issue where isotonic calibration compressed probabilities low enough that a fixed 0.5 threshold predicted "negative" for nearly every image.

## 6. Results

| Metric | Score | Notes |
| --- | --- | --- |
| Mean AUROC (5 conditions) | 0.609 | Range: 0.565 (Nodule) to 0.653 (Atelectasis) |
| Mean Balanced Accuracy | 60.4% | Computed using per-condition tuned thresholds (0.05–0.15), not default 0.5 |

**Per-condition breakdown** (validation set):

| Condition | Threshold Used | Accuracy | Balanced Accuracy | AUROC |
| --- | --- | --- | --- | --- |
| Atelectasis | 0.10 | 56.0% | 63.1% | 0.653 |
| Effusion | 0.10 | 52.0% | 60.9% | 0.638 |
| Infiltration | 0.15 | 33.8% | 58.1% | 0.589 |
| Nodule | 0.05 | 19.6% | 53.6% | 0.565 |
| Pneumothorax | 0.05 | 70.3% | 57.2% | 0.599 |

Predictions file: submitted as `predictions.csv` inside `/predictions`, containing one probability column per condition (Atelectasis, Effusion, Infiltration, Nodule, Pneumothorax) alongside the test set's image ID column, per the format specified in the problem brief.

## 7. Key Insight

The most valuable thing we uncovered was catching how misleading raw accuracy is under class imbalance, and using that to fix our own evaluation process. Early on, a fixed 0.5 threshold made every condition's model appear to predict "negative" almost universally, producing deceptively high raw accuracy (e.g., ~94% for Nodule) while the model was unable to actually distinguish positive cases — a flat 50% balanced accuracy exposed this immediately. Tuning per-condition thresholds against balanced accuracy revealed the models were learning real, if modest, signal all along (AUROC 0.57–0.65), just at probability ranges far below 0.5 given how rare each condition is. This taught us that for imbalanced medical screening tasks, the choice of *evaluation threshold* can matter as much as the choice of model — a well-performing model can look broken, or a broken model can look excellent, purely based on how you choose to threshold and which metric you headline.

## 8. Wildcard Challenges Attempted

| Challenge | Attempted? (Y/N) | Where to find it in the repo |
| --- | --- | --- |
| Best Visualization | Y | `/Wildcard Challenges/WC_01/` — co-occurrence heatmap showing how often patients with one condition also present with another |
| Most Interpretable Model |  |  |
| Fastest Inference Time |  |  |
| Best Handling of Messy/Incomplete Data |  |  |

## 9. Declaration

We confirm that this submission is original work completed by our team during the official event window (Days 1–28), built solely on the officially released dataset, and does not reuse pre-existing projects or another team's code.

KSHITIJ HARIKRISHNAN
***KSHITIJ HARIKRISHNAN***
