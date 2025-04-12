import pandas as pd
import numpy as np
import datetime
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from fxincome import const


class IndexStrategy(StrategyTemplate):
        
    TOTAL_SIZE = 6e6  # Bond Size. 100 face value per unit size. 6 million size -> 600 million face value.
    CASH_AVAILABLE = 630e6  # If use more than available cash, you need to borrow cash.
    MAX_CASH = 10e8  # Maximum cash for backtrader. Order will be rejected if cash is insufficient.
    TOTAL_POS = 6  # Total units of positions.

    # Fixed parameters to be shown and set in the UI
    parameters = ["TOTAL_SIZE", "CASH_AVAILABLE", "MAX_CASH", "TOTAL_POS"]

    # Variables to be shown in the UI
    variables = [
        "low_percentile",  # Low percentile threshold of spread. Default 25th percentile
        "high_percentile",  # High percentile threshold of spread. Default 75th percentile
        "min_volume",  # Mininum trade volume of a bond to be selected. Default 1 billion
        "lookback_days" # Period of historical data to be used for analysis. Default 3*250 trade days
    ]

    def __init__(    
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ):
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.low_pctl = setting.get("low_percentile", 0.25) 
        self.high_pctl = setting.get("high_percentile", 0.75) 
        self.min_vol = setting.get("min_volume", 1e9) 
        self.lookback_days = setting.get("lookback_days", 3*250) 





