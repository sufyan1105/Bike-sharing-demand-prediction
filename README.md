# 🚲 Bike Sharing Demand Prediction

Predicting hourly bike rental demand for a city bike-share system using scikit-learn regression models, with a focus on **time-series-correct evaluation** and **cyclical feature encoding**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![pandas](https://img.shields.io/badge/pandas-2.x-150458)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Capital Bikeshare (Washington, D.C.) recorded every rental between 2011 and 2012. Given the calendar and weather conditions for a single hour, can we predict how many bikes will be rented?

The problem looks like ordinary tabular regression, but it has three properties that punish a careless approach:

1. **Two columns leak the answer.** `casual` and `registered` sum exactly to the target. Including them produces a flawless, useless model.
2. **Time matters.** The rows are chronological. A random train/test split trains on the future to predict the past, which inflates every metric.
3. **Several features are circular.** Hour 23 and hour 0 are one hour apart, but as integers they sit at opposite ends of the range.

This project handles all three explicitly and measures what each decision costs.

**Headline result:** a Random Forest reaches **R² = 0.893** on a held-out chronological test set (RMSE ≈ 72 rides, MAE ≈ 47 rides), beating a regularised linear baseline by roughly 34 R² points.

---

## 📊 Dataset

[UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

| | |
|---|---|
| File used | `hour.csv` (`day.csv` also included) |
| Rows | 17,379 hourly observations |
| Columns | 17 raw → 20 after engineering |
| Period | 1 Jan 2011 – 31 Dec 2012 (731 days) |
| Target | `cnt` — total rentals in that hour |
| Missing values | None |

### Features

| Column | Description | Handling |
|---|---|---|
| `season` | 1=Spring … 4=Winter | Kept as ordinal integer |
| `yr` | 0=2011, 1=2012 | Kept |
| `mnth`, `hr`, `weekday` | Month, hour, day of week | **Cyclically encoded** (sin/cos) + raw retained |
| `holiday`, `workingday` | Calendar flags | Kept |
| `weathersit` | 1=Clear … 4=Heavy rain | Kept as ordinal integer |
| `temp`, `atemp`, `hum`, `windspeed` | Weather, pre-normalised to [0, 1] | Kept |
| `casual`, `registered` | Rider-type counts | **Dropped — target leakage** |
| `instant`, `dteday` | Row index, date string | Dropped |

### A small discovery in the raw data

`cnt` has a minimum of **1**, never 0 — implausible for 4 a.m. in a D.C. January. Comparing row counts explains it:

```
731 days × 24 hours = 17,544 expected
                      17,379 actual
                         165 hours absent
```

Zero-demand hours were never recorded as rows. Worth knowing, because it means "no missing values" is not the same as "no missing data", and a chronological split by row *position* is not exactly a split by *time*.

---

## 🔧 Approach

### Feature engineering

**Cyclical encoding.** Each circular feature becomes a (sin, cos) pair mapping the cycle onto a unit circle, so the last value sits adjacent to the first:

```python
for col, period in [('hr', 24), ('mnth', 12), ('weekday', 7)]:
    df_clean[f'{col}_sin'] = np.sin(2 * np.pi * df_clean[col] / period)
    df_clean[f'{col}_cos'] = np.cos(2 * np.pi * df_clean[col] / period)
```

Both components are required — sine alone is identical for hour 6 and hour 18. Raw integers were **kept alongside** the encodings, since tree models split thresholds more cleanly on `hr` directly than on a smooth sinusoid.

**Domain features.**

| Feature | Definition | Rationale |
|---|---|---|
| `rush_hour` | `hr ∈ {7,8,9,16,17,18}` | Hands a linear model the commuter peaks in one column |
| `comfort_index` | `temp × (1 − hum)` | Warm *and* dry is pleasant; warm and humid is not |

### Train/test split — chronological, not random

```python
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

13,903 training hours (early 2011 → mid 2012) and 3,476 test hours (mid 2012 → end 2012). No shuffling: the row order *is* the information.

This split is honest, and it is also hard. The two halves have genuinely different demand levels:

```
y_train mean: 174.64 rides/hour
y_test  mean: 248.75 rides/hour
```

A 42% gap — the programme grew substantially in its second year. The model is asked to predict a busier world than the one it trained on, which is exactly the situation a deployed forecaster faces.

### Models

| Model | Features | Configuration |
|---|---|---|
| Linear Regression | Standardised | — |
| Ridge Regression | Standardised | `alpha=1.0` |
| Random Forest | Raw | `n_estimators=100`, `random_state=42` |
| Gradient Boosting | Raw | `n_estimators=100`, `random_state=42` |

`StandardScaler` was fitted on the training split only and applied to the test split — important here, since the test set is the future. Tree models use the unscaled DataFrame, preserving column names for the importance plot.

---

## 📈 Results

Ranked by RMSLE (lower is better):

| Model | RMSE | MAE | RMSLE | R² |
|---|---|---|---|---|
| **Random Forest** | **72.26** | **47.38** | **0.401** | **0.893** |
| Gradient Boosting | 99.70 | 67.86 | 0.615 | 0.796 |
| Ridge Regression | 146.84 | 106.19 | 1.046 | 0.556 |
| Linear Regression | 146.84 | 106.19 | 1.046 | 0.556 |

RMSE and MAE are in rides/hour. **RMSLE** measures *relative* error, which matters for count data: missing by 20 rides at 3 a.m. (when 5 were expected) is a proportional catastrophe, while the same 20 at 6 p.m. (when 500 were expected) is noise. All four metrics agree on the ranking here.

### Cross-validation

| Scheme | R² |
|---|---|
| `KFold(n_splits=5)` | 0.802 ± 0.094 |
| `TimeSeriesSplit(n_splits=5)` | 0.773 ± 0.138 |

`TimeSeriesSplit` always trains on earlier rows and tests on later ones, growing the training window each fold. Its fold-to-fold standard deviation is noticeably wider (0.138 vs 0.094) because early folds train on very little data, and because each fold's test period differs in how much growth it has to extrapolate.

Both CV figures sit **below** the single-split test R² of 0.893, which is the useful part: the single split happens to give the model the largest possible training window, so it flatters the result. Reporting 0.77 ± 0.14 is the more defensible number.

### Feature importance

Random Forest mean decrease in impurity, top 10:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `hr` | 0.367 |
| 2 | `hr_sin` | 0.149 |
| 3 | `temp` | 0.077 |
| 4 | `atemp` | 0.073 |
| 5 | `yr` | 0.069 |
| 6 | `workingday` | 0.066 |
| 7 | `hr_cos` | 0.052 |
| 8 | `comfort_index` | 0.042 |
| 9 | `rush_hour` | 0.022 |
| 10 | `weathersit` | 0.015 |

Hour of day accounts for roughly **57%** of total importance once `hr`, `hr_sin` and `hr_cos` are added together — and that split across three columns illustrates a known limitation of impurity-based importance: correlated encodings of one underlying fact divide its credit between them, so each looks more modest than the fact is. The same applies to `temp` and `atemp`, which are near-duplicates by construction.

---

## 🔍 Key findings

**1. Correlation is not importance.** `hr` has a near-zero Pearson correlation with `cnt`, yet it is by far the strongest predictor. Demand is bimodal in hour — peaking around 8 a.m. and 5–6 p.m., low at both ends of the day — and a correlation coefficient only measures *monotonic* association. A feature can drive the target completely while correlating with it not at all.

**2. That bimodality is why trees win by so much.** Linear and Ridge land at R² ≈ 0.556 and are functionally identical (Ridge's penalty changes the fourth decimal place). With ~14,000 training rows and 20 features, overfitting was never the binding constraint — *model form* was. No straight line fits a two-humped curve, and a tree isolates "hour is between 7 and 9" in a single split.

**3. The hourly pattern is not the same every day.** The weekday × hour heatmap shows sharp twin commuter spikes Monday–Friday, replaced by a single broad midday hump at weekends. Hour and `workingday` interact, which trees discover by splitting and linear models cannot express without an explicit interaction term.

**4. Which hours are "worst" depends entirely on the metric.**

| Ranked by raw MAE | | Ranked by MAE ÷ mean demand | |
|---|---|---|---|
| 5 p.m. | 111.8 rides | 4 a.m. | 36% |
| 8 a.m. | 100.8 rides | 3 a.m. | 35% |
| 6 p.m. | 94.5 rides | 1 a.m. | 35% |

The evening peak carries the largest absolute error, so it dominates RMSE. But the small hours are wrong by more than a third of their own volume. Which chart matters depends on the decision: rebalancing a fleet follows absolute error, while a rider checking availability at 4 a.m. experiences the relative one.

**5. `yr` makes this model non-deployable as-is.** It ranks 5th in importance, encoding "2012 was busier than 2011" as a binary flag. Asked about 2013, the model has no representation for it and silently extrapolates from a one-bit feature. A genuine forecaster needs a continuous trend term or explicit growth modelling.

---

## 📁 Repository structure

```
.
├── 03_bike_sharing_demand_predictor.ipynb   # Full analysis, 4 phases
├── hour.csv                                 # Hourly data (17,379 rows)
├── day.csv                                  # Daily aggregate (731 rows)
├── README.md
└── .gitignore
```

The notebook is organised in four phases: **Phase 1** loading and inspection, **Phase 2** exploratory analysis (9 visualisations), **Phase 3** preprocessing and feature engineering, **Phase 4** model training and evaluation.

---

## 🚀 Running it

```bash
git clone https://github.com/sufyan1105/Bike-sharing-demand-prediction.git
cd Bike-sharing-demand-prediction

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

jupyter notebook 03_bike_sharing_demand_predictor.ipynb
```

Run cells top to bottom — later cells depend on variables defined earlier. Both CSVs are included, so no download step is needed.

---

## ⚠️ Limitations

- **`cross_val_score(..., cv=5)` does not shuffle.** scikit-learn's default `KFold` for regressors keeps rows in order, so the 0.802 figure above comes from five contiguous blocks, not a random split. It is therefore *not* a clean measurement of how much random splitting would flatter this model — a genuinely shuffled comparison would need `KFold(n_splits=5, shuffle=True, random_state=42)` passed explicitly.
- **Scaling happens before cross-validation**, so the scaler has seen every fold's test rows. The correct fix is a `Pipeline` wrapping scaler and model, passed to `cross_val_score`.
- **No hyperparameter search.** All settings were chosen by hand; `GridSearchCV` over a `TimeSeriesSplit` would likely improve on them.
- **No lag features.** Demand one hour / one day / one week ago is usually the strongest available signal in real forecasting, and none is used here.
- **165 hours are absent** from the data, almost certainly the zero-demand ones, so the model never learns what genuine zero looks like.

## 🎯 Next steps

- [ ] Wrap preprocessing and model in a `Pipeline`; tune with `GridSearchCV` over `TimeSeriesSplit`
- [ ] Try `PoissonRegressor` — a model built for count targets rather than log-transforming and hoping
- [ ] Add lag features (t−1h, t−24h, t−168h), taking care to use only information available at prediction time
- [ ] Model `casual` and `registered` separately and sum the predictions; the two rider types behave almost oppositely by day of week
- [ ] Benchmark XGBoost / LightGBM against Gradient Boosting
- [ ] Replace `yr` with a continuous trend term so the model can extrapolate beyond 2012

---

## 🛠️ Built with

`python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `jupyter`

## 📄 License

MIT — see [LICENSE](LICENSE).

Dataset: Fanaee-T, H. & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence. Provided by the UCI Machine Learning Repository.

---

**Sufyan Arshad Kadiwala** — M.Sc. AI & Robotics, Hof University of Applied Sciences
[Portfolio](https://sufyankadiwala.de) · [GitHub](https://github.com/sufyan1105) · [LinkedIn](https://linkedin.com/in/sufyan-arshad-kadiwala-a35717290)
