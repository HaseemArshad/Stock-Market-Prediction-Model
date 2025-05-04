# 📈 Stock Market Prediction Model (Real-Time + Historical)

This project leverages the **XGBoost** Machine Learning Model to predict **Close Price** and **MACD values** across a select group of major tech stocks (the Magnificent 7). It starts by exploring both **real-time** and **historical** data separately and then merging our analysis into a **unified dataset** to train a robust and generalizable model capable of capturing both short-term momentum and long-term trends. It showcases the end-to-end steps: **Extracting -> Transforming -> Loading -> Visualizing -> Machine Learning -> Evaluating**

---

## Project Objective:

To build a reliable and scalable stock market prediction system that utilizes:
- Real-time API data (from Finnhub API)
- Historical financial data (from YahooFinance Library)
- Technical indicators like SMA, EMA, MACD, RSI, ATR, Bollinger Bands, OBV(VWAP)
- XGBoost, a gradient-boosted decision tree model, ideal for noisy, non-linear time series

---

## Github Files Layout:

- `HistoricalData.ipynb` - End to End Historical data analysis (not including real time data, or merged model). This is a backup file in case the main file is not able to be run correctly.
- `StockMarketPredictionModel.ipynb` - Main notebook including historical end to end, realtime end to end, as well as unified model
- `stock_data.db` - Our SQLite3 database which stores real time data so it can be queried outside of Stock Market hours (9:30am-4pm Mon-Fri only).
- `README.md` - Briefing of the project
- `requirements.txt` - containing all dependencies for our environment other than libogen which would need to be installed for devices running on MacOS only (needed for XGboost)

---

## How To Run: Clone the repo or download the full folder and folder contents into Jupyterlab.
1. The main project file is StockMarketPredictionModel.ipynb so please run that.
2. If you are running in an IDE (ex: VSCode) be sure to run the first tile of commented-out code so you can pip install requirements.txt.
3. If you are running in JupyterLab, please run the second code tile, which is commented out, and pip install xgboost, yfinance, finnhub-python, and tensorflow. 
4. Additionally, if you are on macos please uncomment the tile that requires libomp to be run, and run/let it be downloaded (this is required for XGBoost).
---

## Summary of how the unified model works in the end: 

### 1. **Column Standardization**
We simplified all rolling window columns (`sma_5`, `sma_20` → `sma`) and made all column names between historical and realtime data compatabile.

### 2. **Unified Dataset Merge**
Both historical and real-time datasets are merged into one:
- A new `is_historical` column shows if the data was once from historical or real-time.
- Target values being (`target_macd`, `target_close`) are made again using shifted values and rolling averages.

### 3. **Training XGBoost**
- Features and targets are scaled using `StandardScaler`.
- Data is split (80/20) and fed into an **XGBoost Regressor**.
- Separate models are trained for **MACD** and **Close Price**.

---

## Quick Overview of our results:

| Target      | MSE     | R²     |
|-------------|---------|--------|
| **MACD**    | ~3.0    | ~0.97  |
| **Close Price** | ~100    | ~0.99  |

- The model **captures turning points, trends, and short-term volatility**.
- The model runs **better than LSTM** due to lack of long sequential data in real-time and because XGBoost handles feature-based time series more effectively.

   
