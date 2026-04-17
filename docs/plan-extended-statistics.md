# Plan: Erweiterte Metriken fuer `stamp statistics`

## Prinzip: Minimal-invasiv

Keine Aenderungen an bestehenden Dateien (`categorical.py`, `roc.py`, `prc.py`).
Stattdessen **neue Dateien hinzufuegen** und in `__init__.py` nach den
bestehenden Aufrufen einfach die neuen Funktionen ausfuehren.

Die bestehende Ausgabe (`individual.csv`, `aggregated.csv`, ROC/PRC Plots)
bleibt exakt wie sie ist.

---

## Neue Dateien

### 1. `src/stamp/statistics/extended_categorical.py`

Liest die gleichen `patient-preds.csv` Dateien wie `_categorical()`,
berechnet aber alle fehlenden Metriken.

**Funktion:** `compute_extended_stats_(pred_csvs, output_dir, ground_truth_label)`

Pro Fold berechnet (aus y_true Labels + y_pred Probability-Matrix):
- macro AUROC, weighted AUROC
- macro F1, weighted F1
- macro AP, weighted AP
- balanced accuracy
- per-class precision, recall, F1
- MCC (Matthews Correlation Coefficient)
- Cohen's Kappa
- aggregierter Log Loss (NLL)
- Confusion Matrix

Aggregiert ueber Folds:
- Mean, SD, 95% CI fuer alle Metriken
- Per-Fold Klassenverteilung (true counts + predicted counts)

**Output:**
- `{label}_extended-stats_individual.csv` — alle Metriken pro Fold
- `{label}_extended-stats_aggregated.csv` — Mean/SD/CI
- `{label}_confusion_matrix_fold-{i}.csv` — Confusion Matrix pro Fold
- `{label}_fold_class_distribution.csv` — true/predicted counts pro Fold

### 2. `src/stamp/statistics/calibration.py`

Liest die gleichen CSVs.

**Funktion:** `compute_calibration_stats_(pred_csvs, output_dir, ground_truth_label)`

Pro Fold:
- Brier Score (multiclass via one-vs-rest)
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)

Aggregiert + Plots:
- Mean/SD/CI fuer Brier, ECE, MCE
- Calibration Plots (Reliability Diagrams) pro Klasse — One-vs-Rest

**Output:**
- `{label}_calibration-stats_individual.csv`
- `{label}_calibration-stats_aggregated.csv`
- `{label}_calibration_plot_{class}.svg` — Reliability Diagram

### 3. `src/stamp/statistics/oof.py`

OOF = Out-of-Fold. Fuegt alle Fold-Predictions zusammen und rechnet globale Metriken.

**Funktion:** `compute_oof_stats_(pred_csvs, output_dir, ground_truth_label)`

1. Liest alle `patient-preds.csv` (split-0 bis split-N)
2. Fuegt sie zusammen (jeder Patient kommt genau 1x vor — aus seinem Test-Fold)
3. Fuegt `fold` Spalte hinzu (aus Datei-Pfad: `split-{i}`)
4. Speichert zusammengefuegtes CSV mit allen Rohdaten
5. Berechnet globale Metriken auf dem OOF-Satz

**Output:**
- `{label}_oof_predictions.csv` — alle Predictions mit fold-ID
- `{label}_oof_stats.csv` — globale Metriken (gleiche wie extended)
- `{label}_oof_confusion_matrix.csv`

---

## Aenderungen an bestehenden Dateien

### `src/stamp/statistics/__init__.py`

**Nur Ergaenzung** — nach den bestehenden Aufrufen von `categorical_aggregated_()`
werden die neuen Funktionen aufgerufen:

```python
# Bestehender Code (unveraendert):
categorical_aggregated_(preds_csvs=pred_csvs, outpath=output_dir, ...)

# NEU — danach:
from stamp.statistics.extended_categorical import compute_extended_stats_
from stamp.statistics.calibration import compute_calibration_stats_
from stamp.statistics.oof import compute_oof_stats_

compute_extended_stats_(pred_csvs=pred_csvs, output_dir=output_dir, ...)
compute_calibration_stats_(pred_csvs=pred_csvs, output_dir=output_dir, ...)
compute_oof_stats_(pred_csvs=pred_csvs, output_dir=output_dir, ...)
```

### Keine Aenderung an:
- `categorical.py` — bleibt wie es ist
- `roc.py` — bleibt
- `prc.py` — bleibt
- `deploy.py` — patient-preds.csv Format bleibt
- `crossval.py` — Fold-Ausgabe bleibt
- Training-Code — nicht betroffen

---

## Implementation Details

### `extended_categorical.py` — Kernlogik

```python
def _compute_fold_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray,
                          categories: list[str]) -> dict:
    y_pred_labels = categories[y_pred_probs.argmax(axis=1)]
    
    return {
        "macro_auroc": roc_auc_score(y_true_onehot, y_pred_probs, average="macro"),
        "weighted_auroc": roc_auc_score(y_true_onehot, y_pred_probs, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred_labels, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred_labels, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_labels),
        "macro_ap": average_precision_score(y_true_onehot, y_pred_probs, average="macro"),
        "weighted_ap": average_precision_score(y_true_onehot, y_pred_probs, average="weighted"),
        "mcc": matthews_corrcoef(y_true, y_pred_labels),
        "cohens_kappa": cohen_kappa_score(y_true, y_pred_labels),
        "log_loss": log_loss(y_true, y_pred_probs),
        # Per-class: precision, recall, f1
        **per_class_metrics,
    }
```

Alle Metriken aus `sklearn.metrics` — keine neuen Abhaengigkeiten (sklearn ist bereits Dependency).

### `calibration.py` — Kernlogik

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def _compute_calibration(y_true_binary, y_pred_probs, n_bins=10):
    brier = brier_score_loss(y_true_binary, y_pred_probs)
    fraction_pos, mean_predicted = calibration_curve(y_true_binary, y_pred_probs, n_bins=n_bins)
    ece = np.mean(np.abs(fraction_pos - mean_predicted))
    mce = np.max(np.abs(fraction_pos - mean_predicted))
    return brier, ece, mce, fraction_pos, mean_predicted
```

One-vs-Rest pro Klasse. Calibration Plots mit `matplotlib`.

### `oof.py` — Kernlogik

```python
def compute_oof_stats_(pred_csvs, output_dir, ground_truth_label):
    dfs = []
    for csv_path in pred_csvs:
        df = pd.read_csv(csv_path)
        fold_name = csv_path.parent.name  # "split-0", "split-1", ...
        df["fold"] = fold_name
        dfs.append(df)
    
    oof_df = pd.concat(dfs, ignore_index=True)
    oof_df.to_csv(output_dir / f"{ground_truth_label}_oof_predictions.csv", index=False)
    
    # Compute all extended metrics on the combined set
    metrics = _compute_fold_metrics(...)
```

---

## Output-Struktur (Beispiel)

```
statistics_output/
  # Bestehend (unveraendert):
  HRDarchival_categorical-stats_individual.csv
  HRDarchival_categorical-stats_aggregated.csv
  roc-curve_HRDarchival=BRCAmut.svg
  pr-curve_HRDarchival=BRCAmut.svg
  
  # NEU — Extended:
  HRDarchival_extended-stats_individual.csv
  HRDarchival_extended-stats_aggregated.csv
  HRDarchival_confusion_matrix_split-0.csv
  HRDarchival_confusion_matrix_split-1.csv
  ...
  HRDarchival_fold_class_distribution.csv
  
  # NEU — Calibration:
  HRDarchival_calibration-stats_individual.csv
  HRDarchival_calibration-stats_aggregated.csv
  HRDarchival_calibration_plot_BRCAmut.svg
  HRDarchival_calibration_plot_LOH+.svg
  HRDarchival_calibration_plot_LOH-.svg
  
  # NEU — OOF:
  HRDarchival_oof_predictions.csv
  HRDarchival_oof_stats.csv
  HRDarchival_oof_confusion_matrix.csv
```

---

## Zusammenfassung

| Aenderung | Dateien |
|-----------|---------|
| Neues File | `statistics/extended_categorical.py` |
| Neues File | `statistics/calibration.py` |
| Neues File | `statistics/oof.py` |
| Minimale Ergaenzung | `statistics/__init__.py` (3 Funktionsaufrufe hinzufuegen) |
| Keine Aenderung | `categorical.py`, `roc.py`, `prc.py`, `deploy.py`, Training-Code |
