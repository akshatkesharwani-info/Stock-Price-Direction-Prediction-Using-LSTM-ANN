# Stock Price Direction Prediction Using LSTM/ANN

ML Engineer Track 

10% free and open-source tools only — no paid APIs, no credit card, runs on Google Colab's free tier.

---

## 📌 Overview

Predicting the exact price is nearly impossible, but predicting next-day direction (up/down) with a useful edge is a common quant research exercise. Build an LSTM neural network using engineered technical-indicator features to predict next-day price direction -- fully in free TensorFlow/Keras, with no paid data or compute required.

**Domain:** Quantitative Research

---

## 📊 Dataset

Free daily OHLCV price history via yfinance

**Source:** [https://finance.yahoo.com](https://finance.yahoo.com)

> This script also includes a **safe fallback**: if the real dataset file isn't found next to the
> notebook/script, it automatically generates a small realistic sample dataset with the same column
> names, so the whole pipeline still runs end-to-end even before you've downloaded the real data.

---

## 🛠️ Tech Stack

Python 3 | TensorFlow/Keras (free) | yfinance | scikit-learn | pandas

**Skills demonstrated:** Python, TensorFlow/Keras, Technical Indicators, Time Series

---

## 🎯 What This Project Builds

- A technical indicator feature set (RSI, MACD, moving averages, volatility, volume change)
- A sliding-window sequence generator for LSTM input (e.g. 20-day lookback windows)
- A binary direction label (next-day close higher or lower than today's close)
- An LSTM classifier built in Keras with dropout regularization
- A simple feedforward ANN baseline for comparison against the LSTM
- Backtested directional accuracy plus a simple long/flat strategy return comparison

---

## 🧭 Step-by-Step Approach

### Step 1: Engineer Features

**What:** Compute RSI, MACD, moving averages, rolling volatility, and volume change

**Why:** Raw price alone is a weak predictor; engineered technical features give the model more signal

**How:** Rolling means/std with pandas; MACD = EMA12 - EMA26


### Step 2: Build Sliding Windows

**What:** Create 20-day lookback sequences of features as LSTM input, with next-day direction as the label

**Why:** LSTMs need sequential windows, not single rows, to learn temporal patterns

**How:** numpy sliding window: X[i] = features[i-20:i], y[i] = 1 if close[i+1]>close[i] else 0


### Step 3: Train LSTM Classifier

**What:** Build a 2-layer LSTM with dropout in Keras and train on the windowed data

**Why:** LSTM layers capture sequential dependencies that a plain ANN on flattened features would miss

**How:** Sequential([LSTM(64, return_sequences=True), Dropout(0.3), LSTM(32), Dense(1, activation='sigmoid')])


### Step 4: Compare to ANN Baseline & Backtest

**What:** Train a simple feedforward ANN on the same flattened features and compare accuracy

**Why:** A baseline is essential to prove the LSTM's sequence-awareness actually adds value

**How:** Sequential([Dense(64,'relu'), Dense(32,'relu'), Dense(1,'sigmoid')]) on flattened last-window features


---

## 📈 Dashboard / Reporting Ideas

- KPI cards: LSTM accuracy vs ANN baseline accuracy, side by side
- Line chart: predicted direction confidence over time overlaid on actual price
- Bar chart: strategy cumulative return vs buy-and-hold over the test period
- Confusion matrix heatmap: up/down prediction accuracy breakdown
- Training curve: loss/accuracy over epochs for both LSTM and ANN

---

## 💡 Key Insights

- Direction prediction accuracy in the low-to-mid 50s% is realistic and still useful for a filtered strategy -- treat any near-70%+ result with skepticism (likely leakage)
- The LSTM vs ANN comparison is the key experiment: it isolates whether sequence memory actually adds predictive value over flat features
- Technical indicators need to be computed strictly on past data at each point -- any lookahead in feature engineering invalidates the whole backtest
- TensorFlow/Keras runs entirely free on CPU for a single-ticker model of this size, no GPU or cloud billing required
- This project is a research exercise in signal quality, not a production trading system -- transaction costs and slippage are excluded

---

## 🚀 How to Run

1. Open the `.py` file in **Google Colab** (free tier — no GPU or paid compute needed) or run it locally with Python 3.
2. Install dependencies with the `pip install ...` line at the top of the script (all free, open-source packages).
3. (Optional) Download the real dataset from the Kaggle link above and place it in the same folder — the filename the script expects is noted in the code's data-loading step. If you skip this, the script auto-generates sample data so you can still see it run.
4. Run the script top to bottom. Outputs (charts, CSVs, model files) are saved in the working directory.

```bash
pip install -r requirements.txt   # or the pip install line at the top of the script
python MLEng_06_Stock_Price_Direction_LSTM_ANN.py
```

---

## 📂 Repo Structure

```
stock-price-direction-prediction-lstm-ann/
├── MLEng_06_Stock_Price_Direction_LSTM_ANN.py       # complete, runnable, free-only solution code
├── README.md              # this file
└── outputs/                # charts, CSVs, and model files generated on run
```

---

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes to demonstrate applied ML/quant-risk
skills. It is not financial, credit, or investment advice, and should not be used for real lending,
trading, or compliance decisions without proper review by a licensed professional.

---

*Part of a 20-project AI Engineer + ML Engineer portfolio focused on finance and consulting use cases —
built entirely with free, open-source tools.*
