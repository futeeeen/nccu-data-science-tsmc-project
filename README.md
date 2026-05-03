# 台積電股票價格預測與交易策略分析系統

本專案依照 `資科_台積電股票價格預測與交易策略分析系統.pdf` 的核心流程實作：

1. 資料蒐集與前處理（`yfinance`）
2. 特徵工程（`MA`, `RSI`, `MACD`）
3. 機器學習分類模型（`Logistic Regression`, `Random Forest`, `XGBoost(可選)`）
4. 模型評估（Accuracy / Precision / Recall / F1 / Confusion Matrix）
5. 交易策略設計與回測（預測上漲才持有）
6. 投資績效評估（Total Return / Max Drawdown / Sharpe Ratio）

重點與 PDF 一致：**模型挑選以回測報酬為主，而不是只看 Accuracy。**

---

## 檔案說明

- `tsmc_stock_system.py`：主程式，從下載資料到回測一次完成
- `backtest_result.csv`：執行後輸出的回測結果（自動產生）

---

## 安裝需求

建議 Python 3.9+。

```bash
pip install pandas numpy yfinance scikit-learn
```

若要啟用 XGBoost 模型：

```bash
pip install xgboost
```

---

## 執行方式

在專案資料夾執行：

```bash
python tsmc_stock_system.py
```

---

## 系統邏輯摘要

1. 下載 `2330.TW`（台積電）歷史資料（預設 2015-01-01 到 2026-01-01）
2. 建立特徵：
   - 基本欄位：`Open`, `High`, `Low`, `Close`, `Volume`
   - 技術指標：`MA5`, `MA20`, `RSI14`, `MACD`, `MACD_signal`, `MACD_hist`
3. 建立預測目標：隔日收盤價上漲為 `1`，否則 `0`
4. 使用時間序列切分（前 80% 訓練，後 20% 測試）避免未來資訊洩漏
5. 比較多個模型的分類表現
6. 以模型預測訊號做策略回測，和 Buy & Hold 比較
7. 輸出績效：
   - `strategy_total_return`
   - `strategy_max_drawdown`
   - `strategy_sharpe`
   - 與 `buy_hold_*` 對照

---

## 輸出結果解讀

- `pred_up=1`：模型預測下一期上漲
- `position=1`：策略持有股票
- `strategy_cum`：策略累積淨值
- `buy_hold_cum`：買入持有累積淨值

若 `strategy_total_return` 長期高於 `buy_hold_total_return`，且回撤可接受，代表策略具備潛在實用價值。

---

## 可延伸方向（報告加分）

1. 加入基本面資料（EPS、ROE）作為特徵
2. 加入交易成本、滑價、停損停利規則
3. 用 Walk-forward / Rolling Window 強化時序驗證
4. 以 Streamlit 製作互動式儀表板

---

## Streamlit 互動式儀表板

安裝：

```bash
pip install streamlit
```

啟動：

```bash
streamlit run streamlit_app.py
```

儀表板功能：

1. 側欄調整股票代號、日期區間、測試集比例、訊號門檻
2. 模型比較表（分類指標 + 策略報酬）
3. 顯示最佳模型（依策略總報酬挑選）
4. 策略 vs Buy & Hold 淨值曲線
5. 下載回測結果 CSV
