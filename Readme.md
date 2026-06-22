# flipkart-gridlock-astram-sense

[![Flipkart GRiD 2.0](https://img.shields.io/badge/Flipkart_GRiD-2.0_Round_2-F4A623)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**Astram Sense** — Predictive event-driven congestion forecasting and resource recommendation system for Bengaluru Traffic Police, built for Flipkart GRiD 2.0 using anonymized Astram operational event data.

**Live Demo:** [astram-sense.streamlit.app](https://astram-sense.streamlit.app)

---

## What This Does

Bengaluru traffic police currently respond to both planned events (rallies, construction, VIP movement) and unplanned ones (breakdowns, accidents, debris) almost entirely on instinct. There is no system to forecast how disruptive an event will be before deploying officers, and no structured way to learn from past responses.

This project closes that gap with a two-part system:

1. **Predicts** how severe an event will be and how long it will last, using patterns in 8,200+ historical Astram events
2. **Recommends** a concrete response — officer count, barricade placement, and diversion route — and logs every prediction against its real outcome so the system improves over time

---

## Key Results

| What | Result |
|---|---|
| Severity classifier (Low / Medium / High / Critical) | **98% validation accuracy**, leak-free, time-based split |
| Duration model | Cause-stratified historical median (293.6 min MAE) outperforms LightGBM and Ridge — documented as an evidence-based finding, not a failure |
| Resource recommendation | Officer count monotonically validated: LOW 2.0 → MEDIUM 4.6 → HIGH 8.0 → CRITICAL 16.2 |
| Feedback loop | Logs predicted vs. actual per event; bias detection by cause/corridor; auto-flags retraining when MAE > 60 min or tier accuracy < 70% |

---

## Repository Structure

```
flipkart-gridlock-astram-sense/
├── Data_Preprocessing_Diagnostic.ipynb   # Full pipeline: cleaning → modeling → rules engine → feedback loop
├── dashboard.py                          # Streamlit app (Live Prediction + Model Performance pages)
├── Astram_Event_Data_Anonymized.csv      # Source dataset (anonymized Bengaluru Traffic Police events)
├── feedback_log.csv                      # Prediction-vs-outcome log (populated at runtime)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Technical Approach

### 1. Data Cleaning
Working with 8,173 anonymized events (Nov 2023 – Apr 2024) from Bengaluru's Astram system, covering ~470 planned events and ~7,700 unplanned. Key issues fixed:
- Outlier durations capped (any event > 24h treated as long-running infrastructure issue, not a traffic event)
- Near-duplicate event reports resolved using rounded-coordinate grouping (441 cases)
- Inconsistent category labels merged (e.g., `Debris`/`debris`)
- `closed_datetime` fallback chain (`closed → resolved → end`) to recover duration for planned events that previously showed NaN
- Rolling-window features computed with a one-row shift to prevent target leakage

### 2. Feature Engineering
- **Time:** IST hour/day cyclically encoded (sin/cos), rush-hour flag, lead time for planned events, `logged_late_flag`
- **Location:** corridor frequency encoding, zone backfill, affected stretch (km from GPS coords), recurring-location event count
- **Historical:** 7-day and 30-day rolling event count and median duration per corridor-cause pair, computed train-only to prevent leakage
- **Text-derived:** description length, redaction-placeholder flags, severity keyword flags
- **Vehicle fields:** missingness indicators (sparse by design, only populated for breakdown events)

### 3. Models
**Severity classifier (LightGBM):**
- Target: combined severity score from priority + road-closure requirement
- Split: train Nov 2023–Feb 2024, validate Mar–Apr 2024 (time-based, not random)
- Leak-prone fields (`priority`, `requires_road_closure`) excluded from feature set
- Result: 98% validation accuracy

**Duration estimation:**
- Three approaches tested: tuned LightGBM, Ridge regression, cause-stratified historical median
- Both ML models perform ~5–7% worse than the naive median at current data volume (~2,400 confirmed-closure training rows)
- Production choice: historical median lookup. ML reserved for when more closure data accumulates.

### 4. Resource Recommendation Engine
Since no historical record of "officers deployed" or "barricades placed" exists in the dataset, a transparent rules layer translates predictions into response:
- **Manpower:** severity tier × road-closure multiplier × corridor-importance weight × stretch-length factor
- **Barricades:** tier-based minimum + spacing scaled to stretch (grounded in IRC:SP:41-2018 guidelines)
- **Diversion:** pre-mapped alternate routes for top recurring corridors/junctions identified by `recurring_loc_count`
- Duration's influence on tier is capped at ±1 level to prevent poor regression predictions from cascading into grossly over/under-allocated responses

---

## Dashboard

Two pages, both crash-free and tested end-to-end:

**Live Prediction**
Enter event cause, corridor, priority, road-closure flag, and affected stretch. One click returns:
- Predicted severity tier (color-coded badge)
- Estimated duration (cause-median lookup)
- Recommended officer count, barricade count, and diversion route

**Model Performance & Bias**
- Predicted vs. actual duration and tier accuracy tracked from `feedback_log.csv`
- Bias broken down by event cause and corridor
- Resource-adequacy feedback from dispatchers
- Retraining trigger status

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Lakshit2604/flipkart-gridlock-astram-sense.git
cd flipkart-gridlock-astram-sense
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the pipeline notebook (optional — regenerates models and feedback log)**
```bash
jupyter notebook Data_Preprocessing_Diagnostic.ipynb
# Run all cells in order
```

**4. Launch the dashboard**
```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

Requirements: Python 3.9+, no GPU needed.

---

## Known Limitations

- **Duration model does not beat the naive baseline** at current data volume. This is documented explicitly, with evidence (two model families, feature importance analysis), and the historical median is used in production as a result.
- **No historical resource-deployment data exists** in the Astram dataset. The recommendation engine uses published traffic-engineering standards (IRC guidelines) rather than learned behavior — this is clearly disclosed.
- **~57% of events lack a usable closure timestamp**, restricting the duration model to confirmed-closure events only. The missingness pattern is systematic (vehicle breakdowns account for 62% of missing rows) and is documented in the notebook.

---

## License

MIT License. See LICENSE for details.

---

## Acknowledgements

- Flipkart GRiD 2.0 and the Astram dataset providers
- Bengaluru Traffic Police for domain context
- IRC:SP:41-2018 (Indian Roads Congress) for barricade-placement guidance

---

*Maintainer: [Lakshit2604](https://github.com/Lakshit2604) — open an issue for questions.*