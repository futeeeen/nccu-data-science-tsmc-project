from __future__ import annotations

import html

import streamlit as st


TERMS = {
    "strategy_mode": {
        "aliases": ["strategy mode", "trading rule"],
        "definition": "策略模式。決定模型分數如何轉成實際持倉或空手規則。",
        "usage": "在本專案中常見為 single_threshold 或 hysteresis。",
        "interpretation": "single_threshold 比較直接；hysteresis 用買入/賣出雙門檻降低頻繁進出。",
    },
    "hysteresis": {
        "aliases": ["dual threshold", "dual threshold buffer", "雙門檻"],
        "definition": "雙門檻緩衝策略。買入門檻與賣出門檻分開，避免分數在門檻附近震盪時一直買賣。",
        "usage": "空手時分數高於買入門檻才進場；持有時分數低於賣出門檻才出場。",
        "interpretation": "適合降低 whipsaw trades，但可能比較晚出場。",
    },
    "single_threshold": {
        "aliases": ["single threshold", "單一門檻"],
        "definition": "單一門檻策略。分數高於門檻就持有，低於門檻就空手。",
        "usage": "用於最簡單的模型訊號回測。",
        "interpretation": "容易理解，但分數在門檻附近震盪時交易次數可能偏多。",
    },
    "selected_buy_threshold": {
        "aliases": ["selected_threshold", "buy_threshold", "selected final score threshold"],
        "definition": "選出的買入門檻。模型分數達到此門檻時，策略才允許進場或持有。",
        "usage": "通常由驗證集 threshold search 選出，或由使用者手動設定。",
        "interpretation": "越高代表越保守，只在模型更有把握時進場。",
    },
    "selected_sell_threshold": {
        "aliases": ["sell_threshold", "selected sell score threshold"],
        "definition": "選出的賣出門檻。雙門檻策略中，持有時分數跌破此門檻才出場。",
        "usage": "搭配 hysteresis 使用。",
        "interpretation": "越低代表願意忍受較長回落；越高代表較快退場。",
    },
    "buy_hold_total_return": {
        "aliases": ["buy & hold return", "buy_hold_return"],
        "definition": "買入持有總報酬。從測試期第一天買入並持有到最後一天的累積報酬。",
        "usage": "作為策略績效的市場基準。",
        "interpretation": "若策略總報酬低於它，代表策略沒有打敗單純持有。",
    },
    "strategy_total_return": {
        "aliases": ["strategy return"],
        "definition": "策略總報酬。依照模型訊號進出場後的累積報酬。",
        "usage": "本專案選模時比 accuracy 更重要，因為交易目標是績效而非只猜對方向。",
        "interpretation": "越高越好，但要同時看最大回撤與 Sharpe Ratio。",
    },
    "holding_ratio": {
        "aliases": ["holding ratio"],
        "definition": "持有比例。測試期間策略有持倉的時間比例。",
        "usage": "用來觀察策略是長時間持有，還是只在少數訊號出現時進場。",
        "interpretation": "太高可能接近 Buy & Hold；太低可能交易機會太少。",
    },
    "entry_count": {
        "aliases": ["entries", "trade count"],
        "definition": "進場次數。策略從空手轉為持有的次數。",
        "usage": "用來評估策略交易頻率。",
        "interpretation": "過高可能交易成本變重；過低可能訊號太保守。",
    },
    "total_cost": {
        "aliases": ["trading cost", "cost_bps"],
        "definition": "總交易成本。依照每次部位變動扣除的成本累計。",
        "usage": "回測中用 bps 模擬手續費、交易稅、滑價等成本。",
        "interpretation": "成本越高，頻繁交易策略越容易失效。",
    },
    "max_drawdown": {
        "aliases": ["strategy_max_drawdown", "buy_hold_max_drawdown", "mdd"],
        "definition": "最大回撤。資產曲線從高點跌到後續低點的最大跌幅。",
        "usage": "衡量策略期間最痛的資金下跌幅度。",
        "interpretation": "越接近 0 越好；大幅負值代表風險較高。",
    },
    "sharpe_ratio": {
        "aliases": ["strategy_sharpe", "buy_hold_sharpe", "sharpe"],
        "definition": "Sharpe Ratio。用平均報酬除以報酬波動，衡量風險調整後績效。",
        "usage": "比較不同策略時，比單看總報酬更能反映穩定性。",
        "interpretation": "越高通常越好，但樣本太短時不一定穩定。",
    },
    "accuracy": {
        "aliases": ["val_accuracy", "test_accuracy"],
        "definition": "準確率。模型預測方向正確的比例。",
        "usage": "分類模型基本指標，但不一定代表策略會賺錢。",
        "interpretation": "金融資料雜訊高，accuracy 高不等於報酬高。",
    },
    "precision": {
        "aliases": ["val_precision"],
        "definition": "精確率。模型預測會上漲的樣本中，實際上漲的比例。",
        "usage": "用來看買入訊號的可信度。",
        "interpretation": "precision 高代表進場訊號比較少誤判。",
    },
    "recall": {
        "aliases": ["val_recall"],
        "definition": "召回率。實際上漲的樣本中，被模型抓到的比例。",
        "usage": "用來看模型是否漏掉很多上漲機會。",
        "interpretation": "recall 高代表比較容易抓到漲勢，但可能也帶來較多誤訊號。",
    },
    "f1": {
        "aliases": ["f1_score", "val_f1"],
        "definition": "F1 分數。precision 與 recall 的綜合指標。",
        "usage": "當你同時在意少誤判與少漏判時使用。",
        "interpretation": "越高代表分類平衡性越好。",
    },
    "target": {
        "aliases": ["label", "y"],
        "definition": "監督式學習的答案標籤。模型訓練時要學習預測的目標。",
        "usage": "技術面與多因子版本多為隔日方向；Triple Barrier 版本則為波段事件標籤。",
        "interpretation": "target 設計會大幅影響模型是否貼近交易目的。",
    },
    "final_score": {
        "aliases": ["score"],
        "definition": "最終分數。把技術面、基本面、籌碼面分數依權重合成後的交易分數。",
        "usage": "策略用它和 threshold 比較，決定是否進場。",
        "interpretation": "越高代表模型越偏向進場。",
    },
    "technical_score": {
        "aliases": ["technical factor"],
        "definition": "技術面分數。由價格、均線、RSI、MACD、成交量等技術指標產生。",
        "usage": "代表技術面目前是否支持進場。",
        "interpretation": "越高通常代表技術面越偏多。",
    },
    "fundamental_score": {
        "aliases": ["fundamental factor"],
        "definition": "基本面分數。由 EPS 與 EPS 成長率等獲利能力資料產生。",
        "usage": "用來衡量公司獲利面是否支持股價。",
        "interpretation": "越高通常代表基本面越強。",
    },
    "chip_score": {
        "aliases": ["chip factor"],
        "definition": "籌碼面分數。由外資、投信、自營商買賣超與融資融券變化產生。",
        "usage": "用來觀察法人與籌碼流向。",
        "interpretation": "越高通常代表籌碼面越偏多。",
    },
    "meta_model": {
        "aliases": ["meta score", "meta_score", "meta_probability"],
        "definition": "第二層模型。用三個因子分數再訓練一個模型，學習它們如何共同影響目標。",
        "usage": "和人工加權分數比較，觀察模型是否能學到更好的組合方式。",
        "interpretation": "不一定永遠比人工加權好，要看驗證與測試績效。",
    },
    "validation": {
        "aliases": ["validation split", "val"],
        "definition": "驗證集。訓練後用來選模型、選 threshold 的資料區段。",
        "usage": "避免直接用測試集調參，降低過度擬合。",
        "interpretation": "驗證集好不代表未來一定好，但比用測試集調參更合理。",
    },
    "test": {
        "aliases": ["test split"],
        "definition": "測試集。保留到最後才看的資料區段，用來評估策略最終表現。",
        "usage": "模擬模型在未知資料上的效果。",
        "interpretation": "測試集績效比訓練/驗證績效更接近真實使用情境。",
    },
    "threshold_search": {
        "aliases": ["auto_threshold", "validation threshold search"],
        "definition": "門檻搜尋。用驗證集嘗試不同買入/賣出門檻，選出回測效果較好的設定。",
        "usage": "讓策略門檻由資料決定，而不是手動猜。",
        "interpretation": "搜尋範圍太大可能過度擬合，需要保留測試集檢查。",
    },
    "buy_threshold": {
        "aliases": ["buy threshold"],
        "definition": "買入門檻。final_score 或 meta_score 達到此數值時才允許進場。",
        "usage": "控制策略進場保守程度。",
        "interpretation": "提高門檻會減少交易次數，降低門檻會增加進場機會。",
    },
    "sell_threshold": {
        "aliases": ["sell threshold"],
        "definition": "賣出門檻。hysteresis 策略中，分數跌破此門檻才出場。",
        "usage": "避免分數小幅下滑就立刻賣出。",
        "interpretation": "低賣出門檻較能抱住波段，但可能承受較大回撤。",
    },
    "buy_hold": {
        "aliases": ["buy and hold", "buy & hold"],
        "definition": "買入持有策略。期初買進，期間不做任何交易，持有到期末。",
        "usage": "用來當策略比較基準。",
        "interpretation": "若模型策略不能穩定勝過它，代表主動交易價值有限。",
    },
    "position": {
        "aliases": ["target_position"],
        "definition": "部位。1 代表持有股票，0 代表空手；部分版本可能支援其他槓桿或做空設定。",
        "usage": "回測用 position 乘上每日報酬計算策略報酬。",
        "interpretation": "觀察 position 可知道策略何時進出場。",
    },
    "strategy_cum": {
        "aliases": ["equity curve", "strategy equity"],
        "definition": "策略累積淨值。把每日策略報酬連乘後形成的資產曲線。",
        "usage": "用圖形觀察策略長期成長與回撤。",
        "interpretation": "曲線越平滑向上越理想。",
    },
    "buy_hold_cum": {
        "aliases": ["buy hold equity"],
        "definition": "買入持有累積淨值。買入持有策略的資產曲線。",
        "usage": "和 strategy_cum 比較，看主動策略是否改善報酬或風險。",
        "interpretation": "策略若只是在大部分時間持有，曲線會接近 buy_hold_cum。",
    },
    "triple_barrier": {
        "aliases": ["triple barrier labeling", "tb_label"],
        "definition": "三重障礙標籤法。用停利、停損、最長持有期三個條件定義交易事件結果。",
        "usage": "Triple Barrier 版本用它取代隔日漲跌標籤。",
        "interpretation": "比隔日漲跌更貼近波段交易，因為它直接描述一筆交易先停利、先停損或時間到。",
    },
    "take_profit": {
        "aliases": ["take_profit_pct", "take-profit barrier"],
        "definition": "停利障礙。進場後價格先上漲到此比例，就標記為成功事件或回測出場。",
        "usage": "Triple Barrier 版本預設為 8%。",
        "interpretation": "設定越高，成功標籤越難出現；設定越低，較容易停利但單筆利潤較小。",
    },
    "stop_loss": {
        "aliases": ["stop_loss_pct", "stop-loss barrier"],
        "definition": "停損障礙。進場後價格先下跌到此比例，就標記為失敗事件或回測出場。",
        "usage": "Triple Barrier 版本預設為 5%。",
        "interpretation": "設定越小，風控越嚴格，但也更容易被短期波動洗出場。",
    },
    "max_holding_days": {
        "aliases": ["vertical_barrier", "vertical barrier"],
        "definition": "最長持有期，也叫垂直時間障礙。若期間內沒有停利或停損，就在此時間點結束事件。",
        "usage": "Triple Barrier 版本用它定義一筆波段交易最多觀察幾天。",
        "interpretation": "越長越偏波段，越短越偏短線。",
    },
}

TRAINING_DATA_TERMS = {
    'Date': {
        'aliases': ['date', 'trading date', 'index date'],
        'definition': 'The trading date for each row in the training dataset.',
        'usage': 'Used to align price, technical, fundamental, and chip features without mixing future information into earlier dates.',
        'interpretation': 'Rows should stay in chronological order because this is a time-series prediction task.',
    },
    'Open': {
        'aliases': ['open price', 'opening price'],
        'definition': 'The first traded price of the asset on that date.',
        'usage': 'Part of the raw OHLCV price data used before feature engineering.',
        'interpretation': 'A large gap between Open and the previous Close can indicate overnight news or market repricing.',
    },
    'High': {
        'aliases': ['high price', 'daily high'],
        'definition': 'The highest traded price of the asset on that date.',
        'usage': "Used with other OHLCV fields to describe the day's price range.",
        'interpretation': 'A high far above Close can mean the price rose intraday but selling pressure appeared later.',
    },
    'Low': {
        'aliases': ['low price', 'daily low'],
        'definition': 'The lowest traded price of the asset on that date.',
        'usage': 'Used with other OHLCV fields to describe downside movement during the day.',
        'interpretation': 'A low far below Close can mean the market recovered after intraday weakness.',
    },
    'Close': {
        'aliases': ['close price', 'closing price'],
        'definition': 'The final traded price of the asset on that date.',
        'usage': 'Most targets and returns are calculated from Close-to-Close price changes.',
        'interpretation': 'Close is the main reference price for next-day direction and backtest returns.',
    },
    'Volume': {
        'aliases': ['trading volume', 'shares traded'],
        'definition': 'The number of shares or units traded on that date.',
        'usage': 'Used to normalize chip flow and identify whether price moves happened with strong participation.',
        'interpretation': 'Higher volume often means the move has stronger market confirmation.',
    },
    'return_1d': {
        'aliases': ['1 day return', 'daily return'],
        'definition': 'The one-day percentage return from the previous Close to the current Close.',
        'usage': 'A short-term momentum feature for the technical model.',
        'interpretation': 'Positive values mean the asset rose from the previous trading day; very high values may signal short-term strength or overextension.',
    },
    'ma_ratio': {
        'aliases': ['moving average ratio', 'price to moving average'],
        'definition': 'A price-to-moving-average ratio used to describe whether price is above or below its recent trend.',
        'usage': 'A technical trend feature for the model.',
        'interpretation': 'Values above 1 usually mean price is above the moving average; values below 1 mean price is below it.',
    },
    'bias_5': {
        'aliases': ['5 day bias', 'five day bias'],
        'definition': 'The short-term deviation of price from its 5-day moving average.',
        'usage': 'Captures short-term overbought or oversold behavior.',
        'interpretation': 'Higher values indicate price is stretched above the short moving average; lower values indicate short-term weakness.',
    },
    'bias_20': {
        'aliases': ['20 day bias', 'twenty day bias'],
        'definition': 'The medium-term deviation of price from its 20-day moving average.',
        'usage': 'Captures medium-term trend distance for the technical model.',
        'interpretation': 'Higher values suggest stronger medium-term trend strength; very extreme values may also mean overheating.',
    },
    'vol_chg': {
        'aliases': ['volume change', 'volume pct change'],
        'definition': 'The recent percentage change in trading volume.',
        'usage': 'Helps the model detect whether market participation is expanding or shrinking.',
        'interpretation': 'Positive values show volume increased; negative values show volume decreased.',
    },
    'rsi_14': {
        'aliases': ['RSI', '14 day RSI', 'relative strength index'],
        'definition': 'A 14-period Relative Strength Index measuring recent upward strength versus downward weakness.',
        'usage': 'Used as a technical momentum and overbought/oversold feature.',
        'interpretation': 'Higher RSI means stronger recent upward momentum; extremely high or low RSI can indicate stretched conditions.',
    },
    'macd': {
        'aliases': ['MACD line'],
        'definition': 'The MACD line, usually calculated from the difference between fast and slow exponential moving averages.',
        'usage': 'A trend and momentum feature for the technical model.',
        'interpretation': 'Higher MACD usually indicates stronger upward momentum relative to the longer trend.',
    },
    'macd_signal': {
        'aliases': ['MACD signal line', 'signal line'],
        'definition': 'The signal line of MACD, usually a moving average of the MACD line.',
        'usage': 'Used together with MACD to detect momentum changes.',
        'interpretation': 'MACD above the signal line is often treated as more bullish than MACD below the signal line.',
    },
    'macd_hist': {
        'aliases': ['MACD histogram', 'histogram'],
        'definition': 'The difference between MACD and its signal line.',
        'usage': 'A technical feature that measures whether momentum is strengthening or weakening.',
        'interpretation': 'Positive values mean MACD is above the signal line; rising values imply improving momentum.',
    },
    'eps': {
        'aliases': ['earnings per share'],
        'definition': 'Earnings per share, a profitability measure from financial statements.',
        'usage': 'Used by the fundamental model after aligning the data to estimated report availability dates.',
        'interpretation': 'Higher EPS generally indicates stronger profitability, but the trend and growth rate are often more useful than the absolute value.',
    },
    'eps_growth_yoy': {
        'aliases': ['EPS YoY', 'year over year EPS growth'],
        'definition': 'Year-over-year EPS growth compared with the same quarter in the previous year.',
        'usage': 'A fundamental feature that measures longer-term earnings improvement or deterioration.',
        'interpretation': 'Positive values mean EPS improved from the same quarter last year; negative values mean it declined.',
    },
    'eps_growth_qoq': {
        'aliases': ['EPS QoQ', 'quarter over quarter EPS growth'],
        'definition': 'Quarter-over-quarter EPS growth compared with the previous quarter.',
        'usage': 'A fundamental feature that captures more recent earnings acceleration or slowdown.',
        'interpretation': 'Positive values mean EPS improved from last quarter; negative values mean recent profitability weakened.',
    },
    'foreign_net_buy': {
        'aliases': ['foreign investor net buy', 'foreign net flow'],
        'definition': 'Daily foreign investor net buying amount or shares.',
        'usage': 'Raw chip-side input that can be transformed into rolling sums and ratios.',
        'interpretation': 'Positive values mean foreign investors bought more than they sold on that day.',
    },
    'investment_trust_net_buy': {
        'aliases': ['investment trust net buy', 'trust net buy'],
        'definition': 'Daily investment trust net buying amount or shares.',
        'usage': 'Raw chip-side input used to measure local institutional fund flow.',
        'interpretation': 'Positive values mean investment trusts bought more than they sold on that day.',
    },
    'dealer_net_buy': {
        'aliases': ['dealer net buy', 'proprietary dealer net buy'],
        'definition': 'Daily dealer net buying amount or shares.',
        'usage': 'Raw chip-side input used to describe dealer-side positioning.',
        'interpretation': 'Positive values mean dealers bought more than they sold on that day.',
    },
    'foreign_net_buy_5d': {
        'aliases': ['5 day foreign net buy', 'foreign 5d net buy'],
        'definition': 'Five-trading-day rolling sum of foreign investor net buying.',
        'usage': 'A chip feature that smooths daily foreign flow into a short-term trend.',
        'interpretation': 'Higher positive values indicate persistent foreign buying over the recent five trading days.',
    },
    'investment_trust_net_buy_5d': {
        'aliases': ['5 day investment trust net buy', 'investment trust 5d net buy'],
        'definition': 'Five-trading-day rolling sum of investment trust net buying.',
        'usage': 'A chip feature that captures whether investment trusts have recently accumulated or sold the asset.',
        'interpretation': 'Higher positive values mean investment trusts were net buyers over the recent five trading days.',
    },
    'dealer_net_buy_5d': {
        'aliases': ['5 day dealer net buy', 'dealer 5d net buy'],
        'definition': 'Five-trading-day rolling sum of dealer net buying.',
        'usage': 'A chip feature that smooths dealer flow over a short recent window.',
        'interpretation': 'Positive values indicate dealers were net buyers over the recent five trading days.',
    },
    'total_institutional_net_buy_5d': {
        'aliases': ['5 day total institutional net buy', 'institutional 5d net buy'],
        'definition': 'Five-trading-day rolling sum of foreign, investment trust, and dealer net buying combined.',
        'usage': 'A chip feature summarizing overall institutional buying pressure.',
        'interpretation': 'Positive values mean institutions as a group bought more than they sold recently.',
    },
    'total_institutional_net_buy_20d': {
        'aliases': ['20 day total institutional net buy', 'institutional 20d net buy'],
        'definition': 'Twenty-trading-day rolling sum of total institutional net buying.',
        'usage': 'A chip feature that captures a longer institutional flow trend.',
        'interpretation': 'Positive values suggest institutions accumulated the asset over about one trading month.',
    },
    'foreign_net_buy_5d_ratio': {
        'aliases': ['foreign 5d ratio', 'foreign net buy volume ratio'],
        'definition': 'Foreign investor 5-day net buy divided by recent trading volume.',
        'usage': 'Normalizes foreign flow so different volume environments are more comparable.',
        'interpretation': 'Higher positive ratios mean foreign buying is large relative to recent volume.',
    },
    'investment_trust_net_buy_5d_ratio': {
        'aliases': ['investment trust 5d ratio', 'trust net buy volume ratio'],
        'definition': 'Investment trust 5-day net buy divided by recent trading volume.',
        'usage': 'Normalizes investment trust flow by recent market activity.',
        'interpretation': 'Higher positive ratios mean investment trust buying is meaningful relative to volume.',
    },
    'dealer_net_buy_5d_ratio': {
        'aliases': ['dealer 5d ratio', 'dealer net buy volume ratio'],
        'definition': 'Dealer 5-day net buy divided by recent trading volume.',
        'usage': 'Normalizes dealer flow so the model can compare it across time.',
        'interpretation': 'Higher positive ratios mean dealer buying is large relative to recent volume.',
    },
    'total_institutional_net_buy_5d_ratio': {
        'aliases': ['institutional 5d ratio', 'total institutional volume ratio'],
        'definition': 'Total institutional 5-day net buy divided by recent trading volume.',
        'usage': 'A normalized chip feature for total institutional buying pressure.',
        'interpretation': 'Higher positive ratios show stronger combined institutional demand relative to trading volume.',
    },
    'foreign_consecutive_buy_days': {
        'aliases': ['foreign consecutive buying days'],
        'definition': 'The number of consecutive recent days with positive foreign investor net buying.',
        'usage': 'A chip feature that detects persistent foreign accumulation.',
        'interpretation': 'Higher values indicate foreign investors have been buying for several days in a row.',
    },
    'investment_trust_consecutive_buy_days': {
        'aliases': ['investment trust consecutive buying days'],
        'definition': 'The number of consecutive recent days with positive investment trust net buying.',
        'usage': 'A chip feature that detects persistent local institutional accumulation.',
        'interpretation': 'Higher values indicate investment trusts have been buying for several days in a row.',
    },
    'margin_balance': {
        'aliases': ['margin financing balance'],
        'definition': 'The outstanding margin financing balance.',
        'usage': 'Raw chip or market leverage data that can be converted into changes over time.',
        'interpretation': 'Rising margin balance can indicate more leveraged buying, which may increase upside participation and downside risk.',
    },
    'short_balance': {
        'aliases': ['short selling balance', 'securities lending balance'],
        'definition': 'The outstanding short balance or securities lending balance.',
        'usage': 'Raw chip or sentiment data that can be converted into changes over time.',
        'interpretation': 'Rising short balance can indicate more bearish positioning or hedging activity.',
    },
    'margin_balance_change_5d': {
        'aliases': ['5 day margin balance change'],
        'definition': 'Five-trading-day change in margin financing balance.',
        'usage': 'A chip feature measuring whether leveraged buying has recently increased or decreased.',
        'interpretation': 'Positive values mean margin financing increased over the recent five trading days.',
    },
    'short_balance_change_5d': {
        'aliases': ['5 day short balance change'],
        'definition': 'Five-trading-day change in short balance.',
        'usage': 'A chip feature measuring whether short-side positioning has recently increased or decreased.',
        'interpretation': 'Positive values mean short balance increased over the recent five trading days.',
    },
    'technical_raw': {
        'aliases': ['raw technical prediction', 'technical raw probability'],
        'definition': "The technical submodel's raw output before calibration to the 0-100 score scale.",
        'usage': 'Shown in the generated score table to separate raw model output from calibrated score.',
        'interpretation': 'Higher values usually mean the technical model sees a higher probability of an upward next-day move.',
    },
    'fundamental_raw': {
        'aliases': ['raw fundamental prediction', 'fundamental raw probability'],
        'definition': "The fundamental submodel's raw output before calibration to the 0-100 score scale.",
        'usage': 'Shown in the generated score table to explain how fundamental data contributes before scoring.',
        'interpretation': 'Higher values usually mean the fundamental model sees more supportive earnings information.',
    },
    'chip_raw': {
        'aliases': ['raw chip prediction', 'chip raw probability'],
        'definition': "The chip submodel's raw output before calibration to the 0-100 score scale.",
        'usage': 'Shown in the generated score table to explain institutional-flow model output before scoring.',
        'interpretation': 'Higher values usually mean the chip model sees stronger supportive flow conditions.',
    },
    'technical_contribution': {
        'aliases': ['technical weighted contribution'],
        'definition': 'The technical score multiplied by its manual strategy weight.',
        'usage': 'Used in the manual weighted strategy to show how much technical analysis contributes to the final score.',
        'interpretation': 'A high value means technical factors are strongly supporting the final manual score.',
    },
    'fundamental_contribution': {
        'aliases': ['fundamental weighted contribution'],
        'definition': 'The fundamental score multiplied by its manual strategy weight.',
        'usage': 'Used in the manual weighted strategy to show how much fundamentals contribute to the final score.',
        'interpretation': 'A high value means EPS and EPS growth features are strongly supporting the final manual score.',
    },
    'chip_contribution': {
        'aliases': ['chip weighted contribution'],
        'definition': 'The chip score multiplied by its manual strategy weight.',
        'usage': 'Used in the manual weighted strategy to show how much institutional flow contributes to the final score.',
        'interpretation': 'A high value means chip-side flow features are strongly supporting the final manual score.',
    },
    'meta_score': {
        'aliases': ['meta model score', 'learned factor score'],
        'definition': "The meta model's learned score based on technical, fundamental, and chip submodel scores.",
        'usage': 'Used to compare a learned factor-combination strategy against the manual weighted strategy.',
        'interpretation': 'Higher values mean the meta model estimates a higher chance of a favorable next-day outcome.',
    },
    'meta_probability': {
        'aliases': ['meta probability', 'meta predicted probability'],
        'definition': "The meta model's predicted probability before or during conversion to a score scale.",
        'usage': 'Used internally to translate the learned relationship among submodel scores into a trading score.',
        'interpretation': 'Higher probabilities indicate the meta model is more confident in a positive target.',
    },
    'factor_data_summary': {
        'aliases': ['factor training overview', 'factor summary'],
        'definition': "A summary table showing each factor group's feature count and usable rows in train, validation, and test splits.",
        'usage': 'Helps verify that technical, fundamental, and chip models have enough clean data for training and evaluation.',
        'interpretation': 'Low usable rows or coverage may make a factor model less reliable.',
    },
    'feature_count': {
        'aliases': ['number of features'],
        'definition': 'The number of feature columns used by a model or factor group.',
        'usage': 'Shown in factor summaries to describe model input size.',
        'interpretation': 'More features can provide richer information, but also increase overfitting risk if data is limited.',
    },
    'overall_coverage': {
        'aliases': ['data coverage', 'coverage'],
        'definition': 'The share of rows that contain usable non-missing data for a factor group.',
        'usage': 'Used to judge whether a factor has enough available information across the selected date range.',
        'interpretation': 'Higher coverage means fewer missing values and more stable training data.',
    },
    'train_usable_rows': {
        'aliases': ['training usable rows'],
        'definition': 'The number of usable rows available for training after cleaning missing values.',
        'usage': 'Shown in factor summaries to verify actual training sample size.',
        'interpretation': 'More usable rows generally make model training more stable.',
    },
    'validation_usable_rows': {
        'aliases': ['validation usable rows', 'val usable rows'],
        'definition': 'The number of usable rows available in the validation split after cleaning missing values.',
        'usage': 'Used to select thresholds and compare models without touching the test split.',
        'interpretation': 'Too few validation rows can make threshold selection unstable.',
    },
    'test_usable_rows': {
        'aliases': ['test usable rows'],
        'definition': 'The number of usable rows available in the final test split after cleaning missing values.',
        'usage': 'Used for the final out-of-sample evaluation.',
        'interpretation': 'The test split should be kept separate from training and threshold selection.',
    },
    'tb_label': {
        'aliases': ['triple barrier label', 'barrier label'],
        'definition': 'The Triple Barrier event label: take-profit first, stop-loss first, or timeout.',
        'usage': 'Used to build the training target in the Triple Barrier strategy.',
        'interpretation': 'Positive labels indicate the profit barrier was reached first; negative labels indicate the stop-loss barrier was reached first; neutral labels indicate timeout.',
    },
    'tb_event': {
        'aliases': ['triple barrier event', 'barrier event'],
        'definition': 'The event type that ended the Triple Barrier observation window.',
        'usage': 'Shows whether the observation ended by take-profit, stop-loss, or max holding days.',
        'interpretation': 'This explains why the Triple Barrier target was assigned for that row.',
    },
    'tb_event_date': {
        'aliases': ['barrier event date'],
        'definition': 'The date when the Triple Barrier event was triggered or timed out.',
        'usage': 'Used to audit when each target label became known.',
        'interpretation': 'Later event dates mean the position would have been held longer before the outcome was decided.',
    },
    'tb_event_return': {
        'aliases': ['barrier event return'],
        'definition': 'The return from the event start date to the date when a barrier was touched or the event timed out.',
        'usage': 'Used to inspect the realized move behind each Triple Barrier label.',
        'interpretation': 'Positive values indicate the event ended with a gain; negative values indicate a loss.',
    },
    'tb_holding_days': {
        'aliases': ['barrier holding days', 'holding days'],
        'definition': 'The number of trading days held until the Triple Barrier event ended.',
        'usage': 'Used to inspect whether labels are generated quickly or mostly by timeout.',
        'interpretation': 'Shorter holding days mean a price barrier was hit quickly; values near max_holding_days often mean timeout.',
    },
    'take_profit_pct': {
        'aliases': ['take profit percent', 'profit barrier percent'],
        'definition': 'The upside return barrier used by Triple Barrier labeling.',
        'usage': 'If price reaches this gain first, the event is labeled as a favorable outcome.',
        'interpretation': 'A higher take-profit barrier creates stricter positive labels.',
    },
    'stop_loss_pct': {
        'aliases': ['stop loss percent', 'loss barrier percent'],
        'definition': 'The downside return barrier used by Triple Barrier labeling.',
        'usage': 'If price reaches this loss first, the event is labeled as an unfavorable outcome.',
        'interpretation': 'A tighter stop-loss barrier creates more sensitive negative labels.',
    },
    'threshold_used': {
        'aliases': ['selected threshold used', 'model threshold'],
        'definition': 'The score threshold actually used to convert model scores into trading signals.',
        'usage': 'Shown in model or strategy summaries to clarify whether the threshold was fixed or selected by validation search.',
        'interpretation': 'A higher threshold usually means fewer but more selective buy signals.',
    },
    'val_strategy_total_return': {
        'aliases': ['validation strategy return'],
        'definition': "The strategy's total return on the validation split.",
        'usage': 'Used during model comparison or threshold search before final test evaluation.',
        'interpretation': 'Higher validation return can guide model selection, but it still needs confirmation on the test split.',
    },
    'test_strategy_total_return': {
        'aliases': ['test strategy return'],
        'definition': "The strategy's total return on the final test split.",
        'usage': 'Used as an out-of-sample performance metric after training and validation decisions are fixed.',
        'interpretation': 'Higher test return means the strategy performed better on unseen data, but risk metrics should also be checked.',
    },
    'test_buy_hold_total_return': {
        'aliases': ['test buy and hold return'],
        'definition': 'The Buy & Hold benchmark return on the final test split.',
        'usage': 'Used to compare the strategy against simply holding the asset.',
        'interpretation': 'If strategy return is below this benchmark, the model did not beat passive holding over the test period.',
    },
    'test_strategy_sharpe': {
        'aliases': ['test Sharpe', 'test strategy Sharpe ratio'],
        'definition': "The strategy's risk-adjusted return on the test split.",
        'usage': 'Used to compare performance after considering volatility, not just raw return.',
        'interpretation': 'Higher Sharpe means better return per unit of volatility.',
    },
    'test_entry_count': {
        'aliases': ['test entries', 'test trade count'],
        'definition': 'The number of times the strategy entered a position during the test split.',
        'usage': 'Used to inspect trading frequency and possible transaction-cost pressure.',
        'interpretation': 'Very high entry count can indicate noisy signals and higher cost sensitivity.',
    },
    'test_holding_ratio': {
        'aliases': ['test holding ratio'],
        'definition': 'The percentage of test-period days where the strategy held a position.',
        'usage': 'Used to understand whether the strategy is selective or almost always invested.',
        'interpretation': 'Higher values mean the strategy behaves more like Buy & Hold; lower values mean it is more selective.',
    },
    'val_f1': {
        'aliases': ['validation F1', 'validation f1 score'],
        'definition': 'F1 score on the validation split, balancing precision and recall.',
        'usage': 'Used to compare classification quality before final test evaluation.',
        'interpretation': 'Higher validation F1 means a better balance between catching positive cases and avoiding false positives.',
    },
    'test_f1': {
        'aliases': ['test F1', 'test f1 score'],
        'definition': 'F1 score on the final test split, balancing precision and recall.',
        'usage': 'Used as an out-of-sample classification metric.',
        'interpretation': 'Higher test F1 means the model generalized better in directional classification.',
    },
    'pred_prob_up': {
        'aliases': ['predicted probability up', 'probability of up move'],
        'definition': "The model's predicted probability that the target will be positive or up.",
        'usage': 'Often converted into a 0-100 score before applying trading thresholds.',
        'interpretation': 'Higher values mean the model is more confident in an upward or favorable outcome.',
    },
    'asset_ret': {
        'aliases': ['asset return', 'underlying return'],
        'definition': "The asset's realized return for the period used in backtesting.",
        'usage': 'Used to calculate strategy and benchmark equity curves.',
        'interpretation': 'Positive values mean the asset rose during that period; negative values mean it fell.',
    },
    'strategy_ret': {
        'aliases': ['strategy return per row'],
        'definition': 'The per-period return earned by the strategy after applying the position rule.',
        'usage': 'Accumulated over time to form the strategy equity curve.',
        'interpretation': 'This is usually zero when the strategy is out of the market and follows asset_ret when it is invested, adjusted for costs if enabled.',
    },
    'buy_hold_ret': {
        'aliases': ['buy hold return per row', 'benchmark return per row'],
        'definition': 'The per-period return of the passive Buy & Hold benchmark.',
        'usage': 'Accumulated over time to form the Buy & Hold equity curve.',
        'interpretation': 'Used as the baseline return that the active strategy tries to beat.',
    },
}

# Add training-tab table columns to glossary search without overriding existing curated terms.
for _term, _item in TRAINING_DATA_TERMS.items():
    TERMS.setdefault(_term, _item)

GLOSSARY_LANGUAGE_OVERRIDES = {
    'strategy_mode': {
        'definition_en': 'Strategy mode. The rule that converts model scores into actual holding or cash decisions.',
        'usage_en': 'Common choices in this project are single_threshold and hysteresis.',
        'interpretation_en': 'single_threshold is simpler; hysteresis uses separate buy and sell thresholds to reduce frequent in/out trades.',
    },
    'hysteresis': {
        'definition_en': 'A dual-threshold buffer strategy that separates buy and sell thresholds to avoid trading too often near one cutoff.',
        'usage_en': 'Enter only when score is above the buy threshold; exit only when held position falls below the sell threshold.',
        'interpretation_en': 'Useful for reducing whipsaw trades, but it may exit later when conditions deteriorate.',
    },
    'single_threshold': {
        'definition_en': 'A single-threshold strategy: hold when the score is above the threshold and stay in cash when below it.',
        'usage_en': 'Used for the simplest model-signal backtest.',
        'interpretation_en': 'Easy to understand, but it can trade frequently when scores hover around the threshold.',
    },
    'selected_buy_threshold': {
        'definition_en': 'The selected buy threshold. The strategy only enters or holds when the model score reaches this level.',
        'usage_en': 'Usually selected by validation threshold search or set manually by the user.',
        'interpretation_en': 'Higher values mean a more conservative strategy that enters only when the model is more confident.',
    },
    'selected_sell_threshold': {
        'definition_en': 'The selected sell threshold. In hysteresis mode, the strategy exits only when the held score falls below this level.',
        'usage_en': 'Used together with hysteresis.',
        'interpretation_en': 'Lower values tolerate larger pullbacks; higher values exit more quickly.',
    },
    'buy_hold_total_return': {
        'definition_en': 'Buy & Hold total return. The cumulative return from buying at the start of the test period and holding to the end.',
        'usage_en': 'Used as the market benchmark for strategy performance.',
        'interpretation_en': 'If the strategy return is lower than this, the active strategy did not beat simple holding.',
    },
    'strategy_total_return': {
        'definition_en': 'Strategy total return. The cumulative return generated by following model-based entry and exit signals.',
        'usage_en': 'More important than accuracy for model selection in this project because the goal is trading performance.',
        'interpretation_en': 'Higher is better, but it should be read together with max drawdown and Sharpe Ratio.',
    },
    'holding_ratio': {
        'definition_en': 'Holding ratio. The percentage of the test period where the strategy is invested.',
        'usage_en': 'Shows whether the strategy holds for long periods or enters only on selected signals.',
        'interpretation_en': 'Very high values may behave like Buy & Hold; very low values may mean too few opportunities.',
    },
    'entry_count': {
        'definition_en': 'Entry count. The number of times the strategy moves from cash to holding.',
        'usage_en': 'Used to evaluate trading frequency.',
        'interpretation_en': 'Too many entries can make transaction costs heavy; too few can mean the signal is too conservative.',
    },
    'total_cost': {
        'definition_en': 'Total trading cost. The accumulated cost deducted when positions change.',
        'usage_en': 'Backtests use bps to approximate fees, tax, and slippage.',
        'interpretation_en': 'Higher costs hurt frequent-trading strategies more.',
    },
    'max_drawdown': {
        'definition_en': 'Maximum drawdown. The largest peak-to-trough decline in the equity curve.',
        'usage_en': 'Measures the most painful capital decline during the strategy period.',
        'interpretation_en': 'Closer to 0 is better; large negative values indicate higher risk.',
    },
    'sharpe_ratio': {
        'definition_en': 'Sharpe Ratio. Average return divided by return volatility, measuring risk-adjusted performance.',
        'usage_en': 'Useful for comparing strategies beyond raw return.',
        'interpretation_en': 'Higher is usually better, but short sample periods can make it unstable.',
    },
    'accuracy': {
        'definition_en': 'Accuracy. The percentage of samples where the model predicted the direction correctly.',
        'usage_en': 'A basic classification metric, but it does not guarantee a profitable strategy.',
        'interpretation_en': 'Financial data is noisy; high accuracy does not always mean high return.',
    },
    'precision': {
        'definition_en': 'Precision. Among samples predicted as up, the percentage that actually went up.',
        'usage_en': 'Used to assess how trustworthy buy signals are.',
        'interpretation_en': 'Higher precision means fewer false buy signals.',
    },
    'recall': {
        'definition_en': 'Recall. Among actual up samples, the percentage caught by the model.',
        'usage_en': 'Used to assess whether the model misses many upside opportunities.',
        'interpretation_en': 'Higher recall catches more rallies, but may also create more false signals.',
    },
    'f1': {
        'definition_en': 'F1 score. A combined metric balancing precision and recall.',
        'usage_en': 'Useful when both false positives and missed opportunities matter.',
        'interpretation_en': 'Higher values indicate a better balance in classification.',
    },
    'target': {
        'definition_en': 'The supervised-learning target label that the model learns to predict.',
        'usage_en': 'Technical and multi-factor versions usually use next-day direction; the Triple Barrier version uses event labels.',
        'interpretation_en': 'The target design strongly affects whether the model matches the trading objective.',
    },
    'final_score': {
        'definition_en': 'Final score. The combined trading score after merging technical, fundamental, and chip scores with weights.',
        'usage_en': 'Compared with thresholds to decide whether to enter or hold.',
        'interpretation_en': 'Higher values mean the model leans more toward entering a position.',
    },
    'technical_score': {
        'definition_en': 'Technical score. A score generated from price, moving averages, RSI, MACD, volume, and other technical features.',
        'usage_en': 'Represents whether technical conditions support entry.',
        'interpretation_en': 'Higher values usually indicate a more bullish technical setup.',
    },
    'fundamental_score': {
        'definition_en': 'Fundamental score. A score generated from EPS and EPS growth features.',
        'usage_en': 'Measures whether profitability data supports the stock price.',
        'interpretation_en': 'Higher values usually indicate stronger fundamentals.',
    },
    'chip_score': {
        'definition_en': 'Chip score. A score generated from institutional net buying and margin/short-balance changes.',
        'usage_en': 'Used to observe institutional flow and positioning.',
        'interpretation_en': 'Higher values usually indicate more bullish chip-side conditions.',
    },
    'meta_model': {
        'definition_en': 'A second-layer model that learns from the three factor scores to predict the target.',
        'usage_en': 'Compared with manual weighted scores to see whether learned factor combinations work better.',
        'interpretation_en': 'It is not guaranteed to beat manual weighting; validation and test results decide.',
    },
    'validation': {
        'definition_en': 'Validation split. The post-training data segment used to select models and thresholds.',
        'usage_en': 'Prevents tuning directly on the test set and reduces overfitting.',
        'interpretation_en': 'Good validation results do not guarantee future performance, but are more reasonable than tuning on test data.',
    },
    'test': {
        'definition_en': 'Test split. The final held-out data segment used only for final strategy evaluation.',
        'usage_en': 'Simulates model performance on unseen data.',
        'interpretation_en': 'Test performance is closer to real use than train or validation performance.',
    },
    'threshold_search': {
        'definition_en': 'Threshold search. Testing different buy/sell thresholds on validation data to select stronger settings.',
        'usage_en': 'Lets the strategy threshold be data-driven instead of guessed manually.',
        'interpretation_en': 'A very wide search range may overfit, so the test split is still needed.',
    },
    'buy_threshold': {
        'definition_en': 'Buy threshold. The score level required for final_score or meta_score to enter a position.',
        'usage_en': 'Controls how conservative entries are.',
        'interpretation_en': 'Raising it reduces trades; lowering it increases opportunities.',
    },
    'sell_threshold': {
        'definition_en': 'Sell threshold. In hysteresis mode, the score must fall below this level before exit.',
        'usage_en': 'Prevents selling immediately after a small score decline.',
        'interpretation_en': 'A lower sell threshold holds trends longer but may tolerate larger drawdowns.',
    },
    'buy_hold': {
        'definition_en': 'Buy & Hold strategy. Buy at the beginning and hold until the end without trading.',
        'usage_en': 'Used as the benchmark for comparison.',
        'interpretation_en': 'If the model cannot consistently beat it, active trading value is limited.',
    },
    'position': {
        'definition_en': 'Position. 1 means holding the asset and 0 means cash; some versions may support other exposure rules.',
        'usage_en': 'Backtests multiply position by returns to compute strategy return.',
        'interpretation_en': 'Inspecting position shows when the strategy enters and exits.',
    },
    'strategy_cum': {
        'definition_en': 'Strategy cumulative equity. The cumulative curve formed by compounding strategy returns.',
        'usage_en': 'Used to visualize long-term growth and drawdowns.',
        'interpretation_en': 'A smoother upward curve is generally better.',
    },
    'buy_hold_cum': {
        'definition_en': 'Buy & Hold cumulative equity. The equity curve for passive holding.',
        'usage_en': 'Compared with strategy_cum to see whether active strategy improves return or risk.',
        'interpretation_en': 'If the strategy holds most of the time, its curve may resemble buy_hold_cum.',
    },
    'triple_barrier': {
        'definition_en': 'Triple Barrier labeling. A method that defines an event outcome using take-profit, stop-loss, and max holding period.',
        'usage_en': 'The Triple Barrier version uses it instead of next-day direction labels.',
        'interpretation_en': 'It is closer to swing trading because it describes whether a trade reaches profit, loss, or timeout first.',
    },
    'take_profit': {
        'definition_en': 'Take-profit barrier. If price rises to this percentage first, the event is labeled successful or exits with profit.',
        'usage_en': 'The Triple Barrier version uses it as the upside boundary.',
        'interpretation_en': 'Higher values make successful labels harder to reach; lower values take profit sooner with smaller gains.',
    },
    'stop_loss': {
        'definition_en': 'Stop-loss barrier. If price falls to this percentage first, the event is labeled failed or exits with loss.',
        'usage_en': 'The Triple Barrier version uses it as the downside boundary.',
        'interpretation_en': 'A tighter stop loss controls risk but may get shaken out by short-term noise.',
    },
    'max_holding_days': {
        'definition_en': 'Maximum holding days, also called the vertical time barrier. If neither price barrier is touched, the event ends here.',
        'usage_en': 'Defines how many days a Triple Barrier trade may be observed.',
        'interpretation_en': 'Longer values are more swing-trade oriented; shorter values are more short-term oriented.',
    },
    'Date': {
    },
    'Open': {
    },
    'High': {
    },
    'Low': {
    },
    'Close': {
    },
    'Volume': {
    },
    'return_1d': {
    },
    'ma_ratio': {
    },
    'bias_5': {
    },
    'bias_20': {
    },
    'vol_chg': {
    },
    'rsi_14': {
    },
    'macd': {
    },
    'macd_signal': {
    },
    'macd_hist': {
    },
    'eps': {
    },
    'eps_growth_yoy': {
    },
    'eps_growth_qoq': {
    },
    'foreign_net_buy': {
    },
    'investment_trust_net_buy': {
    },
    'dealer_net_buy': {
    },
    'foreign_net_buy_5d': {
    },
    'investment_trust_net_buy_5d': {
    },
    'dealer_net_buy_5d': {
    },
    'total_institutional_net_buy_5d': {
    },
    'total_institutional_net_buy_20d': {
    },
    'foreign_net_buy_5d_ratio': {
    },
    'investment_trust_net_buy_5d_ratio': {
    },
    'dealer_net_buy_5d_ratio': {
    },
    'total_institutional_net_buy_5d_ratio': {
    },
    'foreign_consecutive_buy_days': {
    },
    'investment_trust_consecutive_buy_days': {
    },
    'margin_balance': {
    },
    'short_balance': {
    },
    'margin_balance_change_5d': {
    },
    'short_balance_change_5d': {
    },
    'technical_raw': {
    },
    'fundamental_raw': {
    },
    'chip_raw': {
    },
    'technical_contribution': {
    },
    'fundamental_contribution': {
    },
    'chip_contribution': {
    },
    'meta_score': {
    },
    'meta_probability': {
    },
    'factor_data_summary': {
    },
    'feature_count': {
    },
    'overall_coverage': {
    },
    'train_usable_rows': {
    },
    'validation_usable_rows': {
    },
    'test_usable_rows': {
    },
    'tb_label': {
    },
    'tb_event': {
    },
    'tb_event_date': {
    },
    'tb_event_return': {
    },
    'tb_holding_days': {
    },
    'take_profit_pct': {
    },
    'stop_loss_pct': {
    },
    'threshold_used': {
    },
    'val_strategy_total_return': {
    },
    'test_strategy_total_return': {
    },
    'test_buy_hold_total_return': {
    },
    'test_strategy_sharpe': {
    },
    'test_entry_count': {
    },
    'test_holding_ratio': {
    },
    'val_f1': {
    },
    'test_f1': {
    },
    'pred_prob_up': {
    },
    'asset_ret': {
    },
    'strategy_ret': {
    },
    'buy_hold_ret': {
    },
}

for _term, _fields in GLOSSARY_LANGUAGE_OVERRIDES.items():
    if _term in TERMS:
        TERMS[_term].update(_fields)


TRAINING_DATA_ZH_OVERRIDES = {
    'Date': {
        "definition_zh": '每列資料的交易日期。',
        "usage_zh": '用來對齊價格、技術面、基本面與籌碼面資料，避免把未來資料合併到較早日期。',
        "interpretation_zh": '資料應保持時間順序，因為這是時間序列預測任務。',
    },
    'Open': {
        "definition_zh": '當天第一筆成交或開盤價格。',
        "usage_zh": '屬於原始 OHLCV 價格資料，會在特徵工程前使用。',
        "interpretation_zh": '若 Open 與前一日 Close 差距很大，可能代表隔夜消息或市場重新定價。',
    },
    'High': {
        "definition_zh": '當天最高成交價格。',
        "usage_zh": '搭配其他 OHLCV 欄位描述當日價格區間。',
        "interpretation_zh": 'High 遠高於 Close 可能代表盤中衝高後出現賣壓。',
    },
    'Low': {
        "definition_zh": '當天最低成交價格。',
        "usage_zh": '搭配其他 OHLCV 欄位描述盤中下跌幅度。',
        "interpretation_zh": 'Low 遠低於 Close 可能代表盤中轉弱後又收復。',
    },
    'Close': {
        "definition_zh": '當天收盤價格。',
        "usage_zh": '多數 target 與報酬率都由收盤價變化計算。',
        "interpretation_zh": 'Close 是隔日方向預測與回測報酬的主要基準價格。',
    },
    'Volume': {
        "definition_zh": '當天成交量。',
        "usage_zh": '用來標準化籌碼流向，也可判斷價格變動是否有成交量確認。',
        "interpretation_zh": '成交量越高，通常代表該價格變動有較多市場參與。',
    },
    'return_1d': {
        "definition_zh": '前一日收盤到當日收盤的一日報酬率。',
        "usage_zh": '技術模型用來捕捉短期動能。',
        "interpretation_zh": '正值代表較前一交易日上漲；極高值可能代表短線強勢或過熱。',
    },
    'ma_ratio': {
        "definition_zh": '價格相對於移動平均線的比例。',
        "usage_zh": '技術模型用來描述價格在近期趨勢線之上或之下。',
        "interpretation_zh": '大於 1 通常代表價格在均線上方；小於 1 代表在均線下方。',
    },
    'bias_5': {
        "definition_zh": '價格相對 5 日均線的短期乖離。',
        "usage_zh": '用來捕捉短線過熱或超跌。',
        "interpretation_zh": '數值越高代表價格越高於短均線；越低代表短線偏弱。',
    },
    'bias_20': {
        "definition_zh": '價格相對 20 日均線的中期乖離。',
        "usage_zh": '用來描述中期趨勢距離。',
        "interpretation_zh": '數值越高代表中期趨勢較強，但極端值也可能代表過熱。',
    },
    'vol_chg': {
        "definition_zh": '近期成交量變化率。',
        "usage_zh": '幫助模型判斷市場參與度正在增加或降低。',
        "interpretation_zh": '正值代表成交量增加；負值代表成交量下降。',
    },
    'rsi_14': {
        "definition_zh": '14 期相對強弱指標 RSI。',
        "usage_zh": '技術模型用來衡量近期上漲力道與下跌力道。',
        "interpretation_zh": '數值越高代表近期上漲動能越強；極高或極低可能代表狀態偏極端。',
    },
    'macd': {
        "definition_zh": 'MACD 線，通常由快慢指數移動平均線差值計算。',
        "usage_zh": '技術模型用來衡量趨勢與動能。',
        "interpretation_zh": 'MACD 越高通常代表相對長期趨勢的上升動能較強。',
    },
    'macd_signal': {
        "definition_zh": 'MACD 的訊號線，通常是 MACD 線的移動平均。',
        "usage_zh": '搭配 MACD 判斷動能變化。',
        "interpretation_zh": 'MACD 高於 signal line 通常比低於 signal line 更偏多。',
    },
    'macd_hist': {
        "definition_zh": 'MACD 與訊號線的差值。',
        "usage_zh": '衡量動能正在增強或減弱。',
        "interpretation_zh": '正值代表 MACD 高於訊號線；持續上升代表動能改善。',
    },
    'eps': {
        "definition_zh": '每股盈餘 EPS，衡量公司獲利能力。',
        "usage_zh": '基本面模型在對齊財報可取得日期後使用。',
        "interpretation_zh": 'EPS 越高通常代表獲利越強，但趨勢與成長率通常比絕對值更重要。',
    },
    'eps_growth_yoy': {
        "definition_zh": 'EPS 年增率，與去年同季相比。',
        "usage_zh": '基本面模型用來衡量較長期的獲利改善或惡化。',
        "interpretation_zh": '正值代表 EPS 較去年同季成長；負值代表衰退。',
    },
    'eps_growth_qoq': {
        "definition_zh": 'EPS 季增率，與上一季相比。',
        "usage_zh": '基本面模型用來捕捉近期獲利加速或放緩。',
        "interpretation_zh": '正值代表 EPS 較上一季改善；負值代表近期獲利轉弱。',
    },
    'foreign_net_buy': {
        "definition_zh": '外資單日買賣超。',
        "usage_zh": '籌碼面原始輸入，可轉換成 rolling sum 或 ratio。',
        "interpretation_zh": '正值代表外資當日買超，負值代表賣超。',
    },
    'investment_trust_net_buy': {
        "definition_zh": '投信單日買賣超。',
        "usage_zh": '籌碼面原始輸入，用來衡量本土法人資金流。',
        "interpretation_zh": '正值代表投信當日買超，負值代表賣超。',
    },
    'dealer_net_buy': {
        "definition_zh": '自營商單日買賣超。',
        "usage_zh": '籌碼面原始輸入，用來描述自營商部位變化。',
        "interpretation_zh": '正值代表自營商當日買超，負值代表賣超。',
    },
    'foreign_net_buy_5d': {
        "definition_zh": '外資近 5 個交易日買賣超合計。',
        "usage_zh": '把每日外資流向平滑成短期趨勢。',
        "interpretation_zh": '正值越高代表外資近期連續或大量買進。',
    },
    'investment_trust_net_buy_5d': {
        "definition_zh": '投信近 5 個交易日買賣超合計。',
        "usage_zh": '用來觀察投信近期是否累積買進或賣出。',
        "interpretation_zh": '正值越高代表投信近期買盤越強。',
    },
    'dealer_net_buy_5d': {
        "definition_zh": '自營商近 5 個交易日買賣超合計。',
        "usage_zh": '用來平滑自營商短期流向。',
        "interpretation_zh": '正值代表自營商近 5 日偏買方。',
    },
    'total_institutional_net_buy_5d': {
        "definition_zh": '三大法人近 5 個交易日買賣超合計。',
        "usage_zh": '彙整外資、投信、自營商的整體法人買盤。',
        "interpretation_zh": '正值代表法人整體近期偏買方。',
    },
    'total_institutional_net_buy_20d': {
        "definition_zh": '三大法人近 20 個交易日買賣超合計。',
        "usage_zh": '捕捉約一個月的法人流向趨勢。',
        "interpretation_zh": '正值代表法人約一個月內偏累積。',
    },
    'foreign_net_buy_5d_ratio': {
        "definition_zh": '外資 5 日買賣超除以近期成交量。',
        "usage_zh": '把外資流向標準化，方便跨期間比較。',
        "interpretation_zh": '正值越高代表外資買盤相對成交量越明顯。',
    },
    'investment_trust_net_buy_5d_ratio': {
        "definition_zh": '投信 5 日買賣超除以近期成交量。',
        "usage_zh": '把投信流向標準化。',
        "interpretation_zh": '正值越高代表投信買盤相對成交量越有意義。',
    },
    'dealer_net_buy_5d_ratio': {
        "definition_zh": '自營商 5 日買賣超除以近期成交量。',
        "usage_zh": '把自營商流向標準化。',
        "interpretation_zh": '正值越高代表自營商買盤相對成交量越大。',
    },
    'total_institutional_net_buy_5d_ratio': {
        "definition_zh": '三大法人 5 日買賣超除以近期成交量。',
        "usage_zh": '衡量整體法人買盤相對市場成交量的強度。',
        "interpretation_zh": '正值越高代表法人需求相對成交量越強。',
    },
    'foreign_consecutive_buy_days': {
        "definition_zh": '外資連續買超天數。',
        "usage_zh": '偵測外資是否持續累積。',
        "interpretation_zh": '數值越高代表外資已連續多日買超。',
    },
    'investment_trust_consecutive_buy_days': {
        "definition_zh": '投信連續買超天數。',
        "usage_zh": '偵測投信是否持續累積。',
        "interpretation_zh": '數值越高代表投信已連續多日買超。',
    },
    'margin_balance': {
        "definition_zh": '融資餘額。',
        "usage_zh": '籌碼或槓桿資料，可轉換成一段期間的變化。',
        "interpretation_zh": '融資增加可能代表槓桿買盤升高，也可能增加下跌風險。',
    },
    'short_balance': {
        "definition_zh": '融券或借券賣出餘額。',
        "usage_zh": '籌碼或市場情緒資料，可轉換成一段期間的變化。',
        "interpretation_zh": '融券增加可能代表偏空部位或避險增加。',
    },
    'margin_balance_change_5d': {
        "definition_zh": '近 5 日融資餘額變化。',
        "usage_zh": '衡量槓桿買盤近期增加或減少。',
        "interpretation_zh": '正值代表近 5 日融資增加。',
    },
    'short_balance_change_5d': {
        "definition_zh": '近 5 日融券餘額變化。',
        "usage_zh": '衡量偏空部位近期增加或減少。',
        "interpretation_zh": '正值代表近 5 日融券增加。',
    },
    'technical_raw': {
        "definition_zh": '技術子模型校正前的原始輸出。',
        "usage_zh": '用來區分原始模型機率與 0-100 校正分數。',
        "interpretation_zh": '通常越高代表技術模型越認為隔日上漲機率較高。',
    },
    'fundamental_raw': {
        "definition_zh": '基本面子模型校正前的原始輸出。',
        "usage_zh": '說明基本面資料在轉成分數前的模型輸出。',
        "interpretation_zh": '通常越高代表基本面模型越支持正向結果。',
    },
    'chip_raw': {
        "definition_zh": '籌碼面子模型校正前的原始輸出。',
        "usage_zh": '說明法人流向模型在轉成分數前的輸出。',
        "interpretation_zh": '通常越高代表籌碼模型越支持正向結果。',
    },
    'technical_contribution': {
        "definition_zh": '技術分數乘上人工設定權重後的貢獻。',
        "usage_zh": '人工加權策略用它說明技術面對 final_score 的貢獻。',
        "interpretation_zh": '數值越高代表技術面越支持最終分數。',
    },
    'fundamental_contribution': {
        "definition_zh": '基本面分數乘上人工設定權重後的貢獻。',
        "usage_zh": '人工加權策略用它說明基本面對 final_score 的貢獻。',
        "interpretation_zh": '數值越高代表 EPS 與 EPS 成長率越支持最終分數。',
    },
    'chip_contribution': {
        "definition_zh": '籌碼分數乘上人工設定權重後的貢獻。',
        "usage_zh": '人工加權策略用它說明籌碼面對 final_score 的貢獻。',
        "interpretation_zh": '數值越高代表法人流向越支持最終分數。',
    },
    'meta_score': {
        "definition_zh": 'Meta model 根據三個因子分數學出的最終分數。',
        "usage_zh": '用來和人工加權策略比較。',
        "interpretation_zh": '越高代表 Meta model 越認為隔日或事件結果有利。',
    },
    'meta_probability': {
        "definition_zh": 'Meta model 預測的機率。',
        "usage_zh": '用來把三個子模型分數轉換成交易分數。',
        "interpretation_zh": '機率越高代表模型越有信心目標為正向。',
    },
    'factor_data_summary': {
        "definition_zh": '因子資料摘要表。',
        "usage_zh": '顯示各因子特徵數、訓練/驗證/測試可用資料筆數。',
        "interpretation_zh": '可用資料太少或 coverage 太低，該因子模型可能較不穩定。',
    },
    'feature_count': {
        "definition_zh": '模型或因子群使用的特徵欄位數。',
        "usage_zh": '用來描述模型輸入規模。',
        "interpretation_zh": '特徵較多資訊較豐富，但資料少時也可能增加過度擬合。',
    },
    'overall_coverage': {
        "definition_zh": '資料覆蓋率，代表非缺失可用資料比例。',
        "usage_zh": '用來判斷某因子在選定期間是否有足夠資料。',
        "interpretation_zh": 'coverage 越高代表缺失越少，訓練越穩定。',
    },
    'train_usable_rows': {
        "definition_zh": '清理缺失值後訓練集可用筆數。',
        "usage_zh": '用來確認實際訓練樣本數。',
        "interpretation_zh": '可用筆數越多，模型訓練通常越穩定。',
    },
    'validation_usable_rows': {
        "definition_zh": '清理缺失值後驗證集可用筆數。',
        "usage_zh": '用來選 threshold 與比較模型。',
        "interpretation_zh": '筆數太少可能讓 threshold selection 不穩定。',
    },
    'test_usable_rows': {
        "definition_zh": '清理缺失值後測試集可用筆數。',
        "usage_zh": '用於最後的樣本外評估。',
        "interpretation_zh": '測試集應與訓練與調參流程分開。',
    },
    'tb_label': {
        "definition_zh": 'Triple Barrier 事件標籤。',
        "usage_zh": '用來建立 Triple Barrier 策略的訓練目標。',
        "interpretation_zh": '正值代表先碰到停利；負值代表先碰到停損；中性代表時間到。',
    },
    'tb_event': {
        "definition_zh": 'Triple Barrier 事件結束類型。',
        "usage_zh": '顯示事件是因停利、停損或最長持有期而結束。',
        "interpretation_zh": '可解釋該列 target 為何被標成該類。',
    },
    'tb_event_date': {
        "definition_zh": 'Triple Barrier 事件觸發或到期日期。',
        "usage_zh": '用來檢查每個標籤何時才會被知道。',
        "interpretation_zh": '日期越晚代表該筆事件持有越久才決定結果。',
    },
    'tb_event_return': {
        "definition_zh": '事件開始到障礙觸發或到期日的報酬。',
        "usage_zh": '用來檢查每個 Triple Barrier 標籤背後的實際漲跌。',
        "interpretation_zh": '正值代表事件以獲利結束；負值代表虧損。',
    },
    'tb_holding_days': {
        "definition_zh": 'Triple Barrier 事件持有天數。',
        "usage_zh": '用來判斷標籤是很快觸發還是多數靠時間到期。',
        "interpretation_zh": '越短代表很快碰到價格障礙；接近 max_holding_days 常代表 timeout。',
    },
    'take_profit_pct': {
        "definition_zh": 'Triple Barrier 的上方停利比例。',
        "usage_zh": '價格先達到此漲幅時標成有利結果。',
        "interpretation_zh": '設定越高，正向標籤越嚴格。',
    },
    'stop_loss_pct': {
        "definition_zh": 'Triple Barrier 的下方停損比例。',
        "usage_zh": '價格先達到此跌幅時標成不利結果。',
        "interpretation_zh": '設定越緊，負向標籤越敏感。',
    },
    'threshold_used': {
        "definition_zh": '實際用來把模型分數轉成交易訊號的門檻。',
        "usage_zh": '說明門檻是固定設定或由驗證集搜尋選出。',
        "interpretation_zh": '門檻越高通常代表買進訊號較少但更保守。',
    },
    'val_strategy_total_return': {
        "definition_zh": '策略在驗證集的總報酬。',
        "usage_zh": '模型比較或 threshold search 時使用。',
        "interpretation_zh": '較高的驗證報酬可作為選模依據，但仍需測試集確認。',
    },
    'test_strategy_total_return': {
        "definition_zh": '策略在測試集的總報酬。',
        "usage_zh": '用於訓練與驗證決策固定後的樣本外評估。',
        "interpretation_zh": '較高代表策略在未知資料表現較好，但仍要搭配風險指標。',
    },
    'test_buy_hold_total_return': {
        "definition_zh": '測試集 Buy & Hold 基準總報酬。',
        "usage_zh": '用來和單純持有比較。',
        "interpretation_zh": '若策略低於此值，代表測試期沒有打敗被動持有。',
    },
    'test_strategy_sharpe': {
        "definition_zh": '策略在測試集的 Sharpe Ratio。',
        "usage_zh": '衡量考慮波動後的風險調整報酬。',
        "interpretation_zh": '越高代表每承擔一單位波動得到的報酬越好。',
    },
    'test_entry_count': {
        "definition_zh": '測試期間策略進場次數。',
        "usage_zh": '用來觀察交易頻率與交易成本壓力。',
        "interpretation_zh": '過高可能代表訊號太吵且成本敏感。',
    },
    'test_holding_ratio': {
        "definition_zh": '測試期間策略持有部位的比例。',
        "usage_zh": '用來理解策略是選擇性進場還是接近長期持有。',
        "interpretation_zh": '越高越接近 Buy & Hold；越低代表越挑訊號。',
    },
    'val_f1': {
        "definition_zh": '驗證集 F1 分數。',
        "usage_zh": '在最終測試前比較分類品質。',
        "interpretation_zh": '越高代表 precision 與 recall 的平衡越好。',
    },
    'test_f1': {
        "definition_zh": '測試集 F1 分數。',
        "usage_zh": '樣本外分類評估指標。',
        "interpretation_zh": '越高代表模型方向分類泛化較好。',
    },
    'pred_prob_up': {
        "definition_zh": '模型預測目標為正向或上漲的機率。',
        "usage_zh": '通常會轉成 0-100 分後再套用交易門檻。',
        "interpretation_zh": '越高代表模型越相信會上漲或結果有利。',
    },
    'asset_ret': {
        "definition_zh": '回測期間標的實際報酬。',
        "usage_zh": '用來計算策略與基準淨值曲線。',
        "interpretation_zh": '正值代表該期上漲；負值代表下跌。',
    },
    'strategy_ret': {
        "definition_zh": '策略套用部位規則後的單期報酬。',
        "usage_zh": '累積後形成策略淨值曲線。',
        "interpretation_zh": '空手時通常為 0；持有時跟隨 asset_ret，並可能扣除交易成本。',
    },
    'buy_hold_ret': {
        "definition_zh": 'Buy & Hold 基準的單期報酬。',
        "usage_zh": '累積後形成 Buy & Hold 淨值曲線。',
        "interpretation_zh": '作為主動策略要比較或超越的基準。',
    },
}

for _term, _fields in TRAINING_DATA_ZH_OVERRIDES.items():
    if _term in TERMS:
        TERMS[_term].update(_fields)

def _localized_text(item: dict, field: str, lang: str) -> str:
    suffix = "zh" if lang == "zh" else "en"
    return item.get(f"{field}_{suffix}") or item.get(field, "")


def _matches(query: str) -> list[tuple[str, dict]]:
    q = query.strip().lower()
    if not q:
        return []
    results = []
    for term, item in TERMS.items():
        candidates = [term, *item.get("aliases", [])]
        if any(q in candidate.lower() for candidate in candidates):
            results.append((term, item))
    return results


def render_sticky_title_glossary(
    title: str,
    caption: str,
    lang: str = "en",
    key_prefix: str = "glossary",
) -> None:
    st.markdown(
        f"""
        <style>
        .strategy-sticky-title {{
            position: fixed;
            top: 2.6rem;
            left: 21rem;
            right: 2rem;
            z-index: 999;
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(226, 232, 240, 0.95);
            padding: 0.7rem min(30rem, 34vw) 0.55rem 0;
        }}
        .strategy-sticky-title h1 {{
            color: #1f2937;
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.15;
            letter-spacing: 0;
            margin: 0;
        }}
        .strategy-sticky-title p {{
            color: #6b7280;
            font-size: 0.85rem;
            margin: 0.45rem 0 0 0;
        }}
        .strategy-sticky-spacer {{
            height: 8.5rem;
        }}
        details:has(.glossary-panel-marker) {{
            position: fixed;
            top: 4.45rem;
            right: 3rem;
            width: min(34rem, 34vw);
            z-index: 1000;
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(203, 213, 225, 0.95) !important;
            border-radius: 0.45rem;
            box-sizing: border-box;
            box-shadow:
                0 8px 20px rgba(15, 23, 42, 0.08);
            overflow: hidden;
        }}
        details:has(.glossary-panel-marker) div[data-testid="stExpanderDetails"] {{
            max-height: calc(100vh - 9.2rem);
            overflow-y: auto;
            overscroll-behavior: contain;
            padding-bottom: 1rem;
        }}
        @media (max-width: 900px) {{
            .strategy-sticky-title {{
                left: 1rem;
                right: 1rem;
                padding-right: 1rem;
            }}
            .strategy-sticky-title h1 {{
                font-size: 1.35rem;
            }}
            .strategy-sticky-title p {{
                font-size: 0.85rem;
            }}
            .strategy-sticky-spacer {{
                height: 10rem;
            }}
            details:has(.glossary-panel-marker) {{
                top: 8.8rem;
                left: 1rem;
                right: 1rem;
                width: auto;
            }}
            details:has(.glossary-panel-marker) div[data-testid="stExpanderDetails"] {{
                max-height: calc(100vh - 13.5rem);
            }}
        }}
        </style>
        <div class="strategy-sticky-title">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(caption)}</p>
        </div>
        <div class="strategy-sticky-spacer"></div>
        """,
        unsafe_allow_html=True,
    )
    render_glossary_search(lang, key_prefix=key_prefix, sticky=True)


def render_glossary_search(lang: str = "en", key_prefix: str = "glossary", sticky: bool = False) -> None:
    if sticky:
        st.markdown('<div class="glossary-sticky-anchor"></div>', unsafe_allow_html=True)
        panel = st.container()
    else:
        _, panel = st.columns([2.5, 1.15])

    labels = {
        "en": {
            "title": "Glossary / 專有名詞搜尋",
            "input": "Enter a term",
            "placeholder": "e.g. holding_ratio, hysteresis, Sharpe",
            "button": "Search",
            "no_match": "No matching glossary term found. Try threshold, return, position, or triple_barrier.",
            "definition": "Definition",
            "usage": "Usage",
            "interpretation": "How to read it",
            "caption": "Search terms, fields, indicators, and strategy concepts shown on the page.",
            "examples": "Examples: ",
        },
        "zh": {
            "title": "專有名詞搜尋 / Glossary",
            "input": "輸入專有名詞",
            "placeholder": "例如：holding_ratio, hysteresis, Sharpe",
            "button": "搜尋",
            "no_match": "找不到完全相符的專有名詞。可以試試 threshold、return、position、triple_barrier。",
            "definition": "定義",
            "usage": "用法",
            "interpretation": "如何解讀",
            "caption": "可搜尋畫面上的欄位、指標與策略名詞。",
            "examples": "範例：",
        },
    }
    ui = labels.get(lang, labels["en"])

    with panel:
        with st.expander(ui["title"], expanded=False):
            if sticky:
                st.markdown('<span class="glossary-panel-marker"></span>', unsafe_allow_html=True)
            query = st.text_input(ui["input"], placeholder=ui["placeholder"], key=f"{key_prefix}_query")
            search = st.button(ui["button"], key=f"{key_prefix}_button")
            if search or query:
                matches = _matches(query)
                if not matches:
                    st.info(ui["no_match"])
                for term, item in matches[:5]:
                    st.markdown(f"### `{term}`")
                    st.markdown(f"**{ui['definition']}**: {_localized_text(item, 'definition', lang)}")
                    st.markdown(f"**{ui['usage']}**: {_localized_text(item, 'usage', lang)}")
                    st.markdown(f"**{ui['interpretation']}**: {_localized_text(item, 'interpretation', lang)}")
            else:
                examples = ["hysteresis", "selected_buy_threshold", "buy_hold_total_return", "holding_ratio"]
                st.caption(ui["caption"])
                st.caption(ui["examples"] + ", ".join(f"`{x}`" for x in examples))
