# 📈 Stock Market Prediction Model (Real-Time + Historical)

This project ultimately leverges the **XGBoost** Machine Learning Model to predict **Close Price** and **MACD values** across a select group of major tech stocks (the Magnificent 7). It leverages **real-time** and **historical** data into a **unified dataset** to train a robust and generalizable model capable of capturing both short-term momentum and long-term trends. It showcases the end to end steps: extracting -> transforming -> loading -> visualizing -> machine learning -> evaluating

---

## Project Objective:

To build a reliable and scalable stock market prediction system that utilizes:
- Real-time API data (from Finnhub API)
- Historical financial data (from YahooFinance Library)
- Technical indicators like SMA, EMA, MACD, RSI, ATR, Bollinger Bands, OBV(VWAP)
- XGBoost, a gradient-boosted decision tree model, ideal for noisy, non-linear time series

---

## Github Files Layout:

- `HistoricalData.ipynb` - End to End Historical data analysis (not including real time data, or merged model)
- `StockMarketPredictionModel.ipynb` - Master notebook including historical end to end, realtime end to end, as well as unified model
- `stock_data.db` - Our SQLite3 database which stores real time data so it can be queried outside of Stock Market hours (9:30am-4pm Mon-Fri only).
- `README.md` - Briefing of the project
- `requirements.txt` - containing all dependencies for our environment other than libogen which would need to be installed for devices running on MacOS only (needed for XGboost)

---

## Summary of how the unified model works in the end: 

### 1. **Column Standardization**
Unifies rolling window columns (`sma_5`, `sma_20` → `sma`) and makes all column names between historical and realtime data compatabile.

### 2. **Unified Dataset Merge**
Both historical and real-time datasets are merged into one:
- An `is_historical` column marks data origin.
- Missing columns are padded with zeros.
- Target values (`target_macd`, `target_close`) are created using shifted values and rolling averages.

### 3. **Training XGBoost**
- Features and targets are scaled using `StandardScaler`.
- Data is split (80/20) and fed into an **XGBoost Regressor**.
- Separate models are trained for **MACD** and **Close Price**.

---

## Quick Overview of our results:

| Target      | MSE (↓) | R² (↑) |
|-------------|---------|--------|
| **MACD**    | ~3.0    | ~0.97  |
| **Close**   | ~100    | ~0.99  |

- The model **captures turning points, trends, and short-term volatility**.
- Performs **better than LSTM** due to lack of long sequential data in real-time and because XGBoost handles feature-based time series more robustly.


## How To Run: Clone the repo and open the notebook
1. If you are running in an IDE (ex: VSCode) be sure to run the pipinstall for requirements.txt
   
