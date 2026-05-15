from __future__ import annotations

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


def render_glossary_search(lang: str = "en", key_prefix: str = "glossary") -> None:
    _, panel = st.columns([2.5, 1.15])
    title = "專有名詞搜尋 / Glossary"
    placeholder = "例如：holding_ratio, hysteresis, Sharpe"
    with panel:
        with st.expander(title, expanded=False):
            query = st.text_input("輸入專有名詞", placeholder=placeholder, key=f"{key_prefix}_query")
            search = st.button("搜尋", key=f"{key_prefix}_button")
            if search or query:
                matches = _matches(query)
                if not matches:
                    st.info("找不到完全相符的專有名詞。可以試試 threshold、return、position、triple_barrier。")
                for term, item in matches[:5]:
                    st.markdown(f"### `{term}`")
                    st.markdown(f"**定義**：{item['definition']}")
                    st.markdown(f"**用法**：{item['usage']}")
                    st.markdown(f"**如何解讀**：{item['interpretation']}")
            else:
                examples = ["hysteresis", "selected_buy_threshold", "buy_hold_total_return", "holding_ratio"]
                st.caption("可搜尋畫面上的欄位、指標與策略名詞。")
                st.caption("範例：" + ", ".join(f"`{x}`" for x in examples))
