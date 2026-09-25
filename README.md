# Seoul Air Quality: Pollution Pattern Mining & PM2.5 Grade Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![sklearn](https://img.shields.io/badge/sklearn-1.2-orange)
![CI](https://github.com/Ljonjon/seoul-air-quality-ml/actions/workflows/ci.yml/badge.svg)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

An end-to-end machine learning study on hourly air-quality measurements from
25 monitoring stations in Seoul (2017-2019, ~650k rows): unsupervised mining of
co-pollution "chemical signatures" plus supervised PM2.5 health-grade warning,
framed as a public-health early-warning problem where **missed extreme
pollution events cost more than false alarms**.

Originally a graduate course final project (Machine Learning, City University of
Macau, Spring 2026, graded team submission); this repository is the revised,
reproducible version described in [Corrections](#corrections-from-the-graded-course-version).

## Pipeline

```
raw CSVs (summary + station meta + instrument log)
  -> cleaning          drop rows with any non-normal instrument status, -1 sensor sentinels, NaNs
  -> features          hour / day-of-week / month, district one-hot, 1-h lag (PM2.5, PM10),
                       6-h rolling mean of PREVIOUS hours (leakage-free)
  -> target            PM2.5 grade: Good <=15 < Normal <=35 < Bad <=75 < VeryBad  (ug/m3)
  -> splits            PRIMARY: chronological 80/20 (train <= 2019-05-19)
                       comparison: random stratified 80/20
  -> descriptive       K-Means (K=3, elbow on 20k sample) + PCA projection
  -> predictive        Decision Tree / Random Forest / HistGradientBoosting
  -> evaluation        accuracy, macro-F1, per-class report, confusion matrices,
                       dangerous-false-negative count for the VeryBad class
```

## Results

Models share identical splits, scalers (fit on train only) and random seeds.

**Primary protocol - chronological split** (test = last 20% of the timeline,
May-Dec 2019; VeryBad = 0.6% of test rows):

| Model | Accuracy | Macro-F1 | VeryBad recall | Missed VeryBad (FN) |
|---|---|---|---|---|
| Decision Tree | 0.8840 | 0.8706 | 0.809 | 139 / 729 |
| Random Forest | 0.8968 | 0.8739 | 0.757 | 177 / 729 |
| **HistGradientBoosting** | **0.9044** | **0.8919** | **0.835** | **120 / 729** |

**Comparison protocol - random stratified split** (the original course protocol;
optimistic because neighbouring hours leak across train/test):

| Model | Accuracy | Macro-F1 | VeryBad recall | Missed VeryBad (FN) |
|---|---|---|---|---|
| Decision Tree | 0.8681 | 0.8651 | 0.844 | 463 / 2,976 |
| Random Forest | 0.8805 | 0.8750 | 0.830 | 505 / 2,976 |
| **HistGradientBoosting** | **0.8871** | **0.8863** | **0.882** | **352 / 2,976** |

HistGradientBoosting wins under both protocols: boosting sequentially corrects
the residual errors on rare, extreme episodes, which is exactly where a
warning system must not fail.

### Baselines: the numbers that make 0.90 mean something

Hourly PM2.5 is enormously autocorrelated, so "the grade is probably still what it
was an hour ago" is a genuinely strong predictor. Every row below uses the same
chronological split, the same grade thresholds and the same 120,995 test rows.
(`analysis/baselines_importance.py`)

| Predictor | Accuracy | Macro-F1 | VeryBad recall | VeryBad precision | Missed VeryBad | Alerts raised |
|---|---|---|---|---|---|---|
| Uniform random | 0.2516 | 0.1994 | 0.237 | 0.006 | 556 / 729 | 30,235 |
| Majority class (always `Normal`) | 0.4307 | 0.1505 | 0.000 | - | 729 / 729 | 0 |
| **Persistence** (repeat hour *t-1*) | 0.8754 | 0.8494 | 0.807 | 0.807 | 141 / 729 | 729 |
| **HistGradientBoosting (this repo)** | **0.9044** | **0.8919** | **0.835** | **0.923** | **120 / 729** | **660** |

So the honest claim is not "+47 points over the majority class" but **+2.9 pt
accuracy / +4.3 pt macro-F1 over persistence** - and the model dominates
persistence on *both* cost axes for the rare class: 21 fewer missed VeryBad hours
(141 -> 120) **and** 90 fewer false alarms (141 -> 51), which lifts alert
precision from 0.807 to 0.923 at 660 alerts instead of 729. A model that only beat
the majority baseline would be worthless in this problem.

Permutation importance (HistGB, same split, mean accuracy drop over 5 repeats):
current **PM10 0.292**, **PM2.5 lag-1 0.248**, PM10 lag-1 0.097, then the
leakage-free `PM2.5_roll6` at 0.013 and Month at 0.003. All 25 district dummies
together move accuracy by < 0.002, i.e. the signal is chemical and temporal rather
than spatial. Contrast with impurity importance, where `PM2.5_roll6` ranks third
(0.182): a smoothed column splits many nodes cheaply but carries almost no
out-of-sample information - a good illustration of why permutation importance is
the one to trust.

### From a score to a decision: working points and calibration

`predict()` is argmax over four grades, which silently fixes the alert threshold.
`analysis/alert_threshold.py` exposes it instead (HistGB, chronological holdout,
729 real VeryBad hours, ~7.25 months):

| Rule (alert if P(VeryBad) >= tau) | Alerts | Per month | VeryBad recall | Precision | Missed |
|---|---|---|---|---|---|
| argmax (what `predict()` does) | 660 | 91 | 0.835 | 0.923 | 120 |
| tau = 0.25 | 766 | 106 | 0.912 | 0.868 | 64 |
| tau = 0.15 | 846 | 117 | 0.944 | 0.813 | 41 |
| tau = 0.10 | 916 | 126 | 0.956 | 0.761 | 32 |
| tau = 0.05 | 1,053 | 145 | 0.973 | 0.673 | 20 |

argmax is *not* the best working point: tau = 0.25 already finds 7.7 pt more of the
rarest class for ~15 extra alerts a month. Each further step has a price - going from
tau 0.25 to 0.15 buys 23 fewer missed hours at the cost of 57 more false ones, so it
only pays if a missed episode is worth at least 2.5x a false alert; 0.15 -> 0.10 needs
6.8x; 0.10 -> 0.05 needs 10.4x. That table is the answer to "how would you operate it".

The raw probabilities turn out to be usable as they are: Brier 0.00101 (vs 0.00599 for
a base-rate predictor) and 10-bin ECE 0.00032. Adding isotonic calibration fitted on
the training block made both *worse* (Brier 0.00105, ECE 0.00086) because it maps onto
a 2.9% training-period base rate while the holdout sits at 0.6% - a clean
demonstration that calibration is not free and has to be fitted on a window that
resembles the deployment period.

### Clustering: three chemical signatures

K-Means on the five standardized pollutants **excluding PM2.5** (kept out so the
descriptive task does not circularize the predictive target). Elbow at K=3;
PCA (PC1 50.0%, PC2 23.3%) shows one dense baseline cloud and two extreme
long-tail regimes:

| Cluster | Share | Signature (z-scores) | Reading |
|---|---|---|---|
| 2 | 52.6% | all pollutants below mean | clean baseline air |
| 1 | 27.7% | NO2 +1.2, CO +1.1, low O3 | traffic / combustion episodes |
| 0 | 19.6% | O3 +1.3, high SO2 & PM10 | photochemical / secondary-pollution events |

## Figures

| Elbow (K selection) | PCA projection of clusters |
|---|---|
| ![elbow](figures/elbow_method.png) | ![pca](figures/pca_clusters.png) |

| Model accuracy (chronological) | Best model confusion matrix |
|---|---|
| ![acc](figures/acc_chrono.png) | ![cm](figures/cm_chrono_histgradientboosting.png) |

## Getting started

1. `pip install -r requirements.txt`
2. Download the *Seoul Air Quality* dataset (Kaggle: "Air Pollution Seoul",
   hourly means, 25 stations, 2017-2019) and place the CSVs as:

```
AirPollutionSeoul/
├── Measurement_summary.csv
└── Original Data/
    ├── Measurement_info.csv
    └── Measurement_station_info.csv
```

3. Run:

```bash
python main.py                 # uses ./AirPollutionSeoul
python main.py --data /path/to/AirPollutionSeoul
```

Runtime ~1 min on a laptop CPU. Outputs: console log
(`results/run_log.txt` via redirection), metrics JSON
(`results/classification_results.json`), cluster profile CSV, and all figures
in `figures/`. Individual stages: `python data_preprocessing.py`,
`python task_clustering.py`, `python task_classification.py`.
`render_cm.py` re-draws confusion heatmaps from the JSON without retraining.

### Tests / CI

```bash
pytest -q          # 10 tests, ~5 s, no dataset required
```

`tests/test_pipeline.py` builds a small synthetic clone of the three source CSVs
(deterministic diurnal pollutant cycle, abnormal-status hours, one `-1` sensor
sentinel) and asserts the leakage / ordering / split / scaling invariants above, so
the methodological fixes cannot silently regress. The raw dataset is not
redistributed here; regenerate `figures/` and `results/` locally with
`python main.py`.

### Analysis scripts: every claim in this README is re-computable

```bash
python analysis/data_audit.py             # -> results/data_audit.json           (~11 s)
python analysis/baselines_importance.py   # -> results/baselines_importance.json (~30 s)
python analysis/pca_variance_check.py     # -> results/pca_variance.json         (~12 s)
python analysis/leak_ablation.py          # -> results/leak_ablation.json        (~6 min)
python analysis/cluster_validity.py       # -> results/cluster_validity.json       (~1 min)
python analysis/alert_threshold.py        # -> results/alert_threshold.json        (~40 s)
```

`main.py` produces the models; these four scripts produce everything else quoted
above. They share `analysis/_common.py` (path resolution, the leaky-vs-fixed feature
builders, both split protocols) so an ablation is a flag flip, not a copy of the
pipeline, and they dump JSON so the numbers can be diffed instead of remembered.

- **data_audit** - grid completeness (647,511 of the 657,000 expected
  hour x station cells), the 9,489 missing cells decomposed into 374 citywide-absent
  hours vs 30 partially-missing hours, instrument-status counts, `-1` sentinel
  counts per pollutant, physically impossible `PM2.5 > PM10` rows (13,439 raw ->
  4,423 after the status filter), the ten worst PM2.5 readings (max 6,256 ug/m3 -
  all ten flagged by the instrument log), and class risk by month/hour.
- **baselines_importance** - the baseline table, permutation importance and the
  per-class report. Writing it caught a bug in my own earlier draft, which reported
  persistence accuracy as 0.47 by binning the *standardized* lag column with the raw
  15/35/75 thresholds; the audited value is 0.8754 (see Corrections).
- **pca_variance_check** - explained variance across 10 scaler x column-set x
  row-set combinations, which reproduces the course report's "82%" as MinMax scaling
  of six pollutants (PC1 = 82.2%) and shows the pipeline's own standardized five
  pollutants at 50.0 / 23.3 / 13.6.
- **leak_ablation** - the 6-cell study below.
- **cluster_validity** - silhouette / Calinski-Harabasz / Davies-Bouldin for
  K = 2..6, plus the reason K = 3 was chosen: PM2.5 is *not* a clustering feature, yet
  the clusters separate it 40.7 / 31.5 / 14.9 ug/m3 and its VeryBad rate
  8.94% / 2.26% / 0.004% against a 2.93% base rate - and the ordering survives on the
  2019 holdout even though the base rate there drops to 0.60%.
- **alert_threshold** - the threshold and calibration tables above.


## Repository layout

```
main.py                     entry point: python main.py [--data PATH] [--k K]
data_preprocessing.py       cleaning, [FIX-1..5] features, both split protocols, scaling
task_clustering.py          elbow study, K-Means, PCA projection, cluster profiles
task_classification.py      DT / RF / HistGB, metrics, figures, results JSON
render_cm.py                re-draw confusion heatmaps from results JSON (no retraining)
tests/test_pipeline.py      synthetic-data regression tests (CI; no dataset needed)
analysis/_common.py           shared loaders: leaky vs fixed features, both splits
analysis/*.py                 data audit, baselines+importance, PCA forensics, ablation
figures/                    elbow, PCA, accuracy bars, 6 confusion matrices (generated)
results/                    metrics + audit JSONs, cluster_profiles.csv, run_log.txt
requirements.txt            minimal pinned dependencies
```

## Corrections from the graded course version

The submitted course version reported 88.54% accuracy under a random split. While
preparing this repository I found and fixed four methodological issues, then added a
regression-test safety net so they cannot come back. Every number above comes from the
corrected pipeline:

1. **Label leakage in `PM2.5_roll6`** - the rolling mean included the *current*
   hour, i.e. the very value the target grade is derived from. Now computed on
   the six *previous* hours only (`shift(1)` before `rolling`).
2. **Random split on time-series data** - neighbouring hours leaked between
   train and test. The primary protocol is now a chronological 80/20 split;
   the random split is kept as an explicit comparison.
3. **Unused instrument-status filter** - the course code loaded
   `Measurement_info.csv` but never applied it; rows where any sensor reported
   a non-normal status (6.6% of the data) are now actually dropped.
4. **Warm-up ordering** - lag/rolling features are now computed after an
   explicit sort by (station, timestamp).
5. **No safety net** - the fixes are pinned by `tests/test_pipeline.py`, a
   synthetic fixture (no dataset download required) that regenerates the three
   source CSVs and asserts: the status filter drops exactly the flagged hours;
   the rolling window excludes the label hour; an *intervention test* where
   PM2.5 at hour t is changed leaves hour t's own features untouched and shifts
   precisely the six following rows; lag/rolling never cross stations; the
   chronological split never trains on the future; scalers are train-fit only.
   GitHub Actions runs it on Python 3.10-3.12 for every push.

Two more errors were caught by the audit scripts rather than by the code, and both
are worth stating out loud:

- An earlier draft quoted the persistence baseline as 0.47 accuracy. That number came
  from `pd.cut()`-ing the *standardized* `PM2.5_lag1` column with the raw 15/35/75
  thresholds, so it degenerated into "always Good". The audited value is **0.8754**,
  which makes the result harder to earn and the claim honest (see Baselines above).
- The course report's "PCA explains 82%" is reproducible only as *Min-Max* scaling of
  six pollutants (PC1 alone = 82.2%); with the standardized five-pollutant input the
  clustering actually uses, PC1/PC2/PC3 = 50.0/23.3/13.6%. An explained-variance
  number without its preprocessing is not a number.

### What each fix is worth

`analysis/leak_ablation.py` re-runs the pipeline over
{leaky, fixed} x {unfiltered, filtered} x {random, chronological} - HistGB:

| Cell | Leak | Status filter | Protocol | Accuracy | Macro-F1 | VeryBad recall | Missed VeryBad |
|---|---|---|---|---|---|---|---|
| E - course version | yes | no | random | 0.8851 | 0.8762 | 0.840 | 547 / 3,414 |
| A | yes | yes | random | 0.8902 | 0.8889 | 0.882 | 352 / 2,977 |
| C | no | no | random | 0.8798 | 0.8724 | 0.828 | 586 / 3,414 |
| B = course protocol, corrected | no | yes | random | 0.8871 | 0.8863 | 0.882 | 352 / 2,976 |
| L | yes | yes | chronological | 0.9087 | 0.8937 | 0.841 | 116 / 729 |
| **D = this repo** | **no** | **yes** | **chronological** | **0.9044** | **0.8919** | **0.835** | **120 / 729** |

Marginal effects: **removing the leak costs 0.31 pt accuracy / 0.26 pt macro-F1**
(A -> B) and 0.43 pt accuracy under the chronological protocol (L -> D) - the
rolling mean carries only 1/6 of the current hour, and the target is nearly a
function of that same hour anyway, so the inflation was real but modest.
**Applying the status filter is worth +0.73 pt accuracy and +5.3 pt VeryBad recall**
(C -> B): the dropped rows are exactly the sensor-fault hours that produced fake
extremes. **Switching to a chronological split raises accuracy by 1.7 pt** while
leaving 729 instead of 2,976 VeryBad hours in test, because the summer holdout is
the cleanest part of the year - which is precisely why the evaluation protocol has
to be reported next to every number.


## Limitations

- Single-cut validation; a rolling-origin evaluation would be stronger.
- The model *nowcasts* the current hour's grade from that hour's co-pollutants plus
  PM2.5 history; a true 1-hour-ahead forecast would additionally have to shift the
  current-hour covariates (see the prediction-horizon discussion in the write-up).
- No meteorological covariates (wind, temperature) - the biggest known driver
  of dispersion; see the report's future-work section.
- K-Means assumes convex clusters; the long-tail regimes suggest GMM or HDBSCAN as
  follow-ups, and the internal indices (silhouette/CH/DB) actually prefer K = 2, which is
  reported rather than hidden - K = 3 was kept for interpretability (see Clustering).
- Thresholds: `analysis/alert_threshold.py` quantifies the precision/recall frontier
  and shows argmax is not the best working point, but tau is still chosen by hand from
  that table; picking it with rolling-origin CV under an explicit alert budget is the
  next step.

## Team

Two-person course team: one member contributed the visualization scheme, model
pipeline and report; the other the experimental analysis and framework.
Personal student IDs and the graded report PDF are intentionally not published
here.

## License

MIT - see [LICENSE](LICENSE). Code, figures and derived metrics are mine; the
underlying *Seoul Air Quality* measurements belong to the public dataset authors
and are not redistributed in this repository.
