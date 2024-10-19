class PATH:
    MAIN = "d:/ProjectRicequant/fxincome/"
    YTM_MODEL = MAIN + "model/ytm/"
    SPREAD_DATA = MAIN + "spread/"
    SPREAD_MODEL = MAIN + "model/spread/"
    STRATEGY_POOL = MAIN + "strategies_pool/"

class MYSQL_CONFIG:
    HOST = "localhost"
    PORT = 3306
    USER ='root'
    PASSWORD=''
    DATABASE='fxincome'
class COUPON_TYPE:
    REGULAR = "附息"
    ZERO = "贴现"  # 贴现债券剩余期限不能超过1年
    DUE = "到期一次还本付息"  # TODO 现在算的到期还本付息债券不能超过1年


class ACCOUNT_TYPE:
    OCI = "OCI"
    TPL = "TPL"
    AC = "AC"


class CASHFLOW_TYPE:
    Undelivered = "Undelivered"
    Undelivered_Lastone = "Undelivered_Lastone"
    History = "History"
    All = "All"


class CASHFLOW_VIEW_TYPE:
    Raw = "Raw"
    Agg = "Agg"


class POSITION_GAIN_VIEW_TYPE:
    Raw = "Raw"
    Agg = "Agg"


class DURARION_TYPE:
    Macaulay = "Macaulay"
    Modified = "Modified"


class TBOND_PARAM:
    """
    Parameters for 16国债19(019547.SH)
    训练模型的参数。模型目标是预测16国债19收盘收益率的涨跌。
    参数主要是Features和Labels
    """

    ALL_FEATS = [
        # 原始features
        "date",
        "close",  # 收盘ytm
        "amount",  # 成交额，单位是元
        "ttm",  # Term to Maturity, 剩余期限，单位是年
        "fr007",  # 7天回购定盘利率
        "t10y",  # 10年国债中债估值ytm
        "fr007_1y",  # 1年期 fr007 IRS CFETS收盘利率
        "fr007_5y",  # 5年期 fr007 IRS CFETS收盘利率
        "usdcny",  # 美元兑人民币汇率
        # 收盘ytm变种
        "pct_chg",
        "avg_chg_5",
        "avg_chg_10",
        "avg_chg_20",
        "volaty",  # (ytm(low price)  - ytm(high price)) / ytm(close)
        # 流动性指标变种
        "fr007_chg_5",  # fr007 定盘利率 前5天均值变化率
        "fr007_1y_chg_5",  # 1年期 fr007 IRS CFETS收盘利率 前5天均值变化率
        # 其他指标变种
        "t10y_chg_5",
        # 收盘ytm与其他各种指标之间的差值
        "spread_t1y",
        "spread_t10y",
        "spread_fr007",
        "spread_fr007_1y",
        "spread_fr007_5y",
        "spread_usdcny",
        # 其他各种指标之间的差值
        "spread_fr007_5y_fr007_1y",  # 5年IRS - 1年IRS
        # 汇率变种
        "usdcny_chg_5",
        # label
        "future",
        "target",
    ]

    TRAIN_FEATS = [
        # 原始features
        "close",
        # 'amount',
        # 'ttm',
        # 'fr007',
        # 'fr007_5y',
        # 收盘ytm变种
        "pct_chg",
        "avg_chg_5",
        "avg_chg_10",
        # 'avg_chg_20',
        # 'volaty',
        # 流动性指标变种
        "fr007_chg_5",
        # 其他指标变种
        # 't10y_chg_5',
        # 收盘ytm与其他各种指标之间的差值
        "spread_t1y",
        # 'spread_t10y',
        "spread_fr007",
        # 'spread_fr007_5y',
        "spread_usdcny",
        # 其他各种指标之间的差值
        # 'spread_fr007_5y_fr007_1y',
        # 汇率变种
        "usdcny_chg_5",
    ]

    NN_TRAIN_FEATS = [
        # 原始features
        "close",  # 收盘ytm
        "amount",  # 成交额，单位是元
        # 'fr007',  # 7天回购定盘利率
        "t10y",  # 10年国债中债估值ytm
        # 'fr007_1y',  # 1年期 fr007 IRS CFETS收盘利率
        "fr007_5y",  # 5年期 fr007 IRS CFETS收盘利率
        # 'usdcny',  # 美元兑人民币汇率
        # 收盘ytm变种
        "pct_chg",
        # 'avg_chg_5',
        "avg_chg_10",
        # 'avg_chg_20',
        # 'volaty',  # (ytm(low price)  - ytm(high price)) / ytm(close)
        # 流动性指标变种
        # 'fr007_chg_5',  # fr007 定盘利率 前5天均值变化率
        # 'fr007_1y_chg_5',  # 1年期 fr007 IRS CFETS收盘利率 前5天均值变化率
        # 其他指标变种
        # 't10y_chg_5',
        # 收盘ytm与其他各种指标之间的差值
        # 'spread_t1y',
        "spread_t10y",
        "spread_fr007",
        # 'spread_fr007_1y',
        "spread_fr007_5y",
        "spread_usdcny",
        # 其他各种指标之间的差值
        # 'spread_fr007_5y_fr007_1y',  # 5年IRS - 1年IRS
        # 汇率变种
        # 'usdcny_chg_5',
    ]

    SCALED_FEATS = [
        # 原始features
        "close",  # 收盘ytm
        "amount",  # 成交额，单位是元
        # 'fr007',  # 7天回购定盘利率
        "t10y",  # 10年国债中债估值ytm
        # 'fr007_1y',  # 1年期 fr007 IRS CFETS收盘利率
        "fr007_5y",  # 5年期 fr007 IRS CFETS收盘利率
        # 'usdcny',  # 美元兑人民币汇率
        # 收盘ytm变种
        # 'pct_chg',
        # 'avg_chg_5',
        # 'avg_chg_10',
        # 'avg_chg_20',
        # 'volaty',  # (ytm(low price)  - ytm(high price)) / ytm(close)
        # 流动性指标变种
        # 'fr007_chg_5',  # fr007 定盘利率 前5天均值变化率
        # 'fr007_1y_chg_5',  # 1年期 fr007 IRS CFETS收盘利率 前5天均值变化率
        # 其他指标变种
        # 't10y_chg_5',
        # 收盘ytm与其他各种指标之间的差值
        # 'spread_t1y',
        # 'spread_t10y',
        # 'spread_fr007',
        # 'spread_fr007_1y',
        # 'spread_fr007_5y',
        "spread_usdcny",
    ]

    FEAT_OUTLINERS = []

    LABELS = ["target"]


class MTM_PARAM:
    """
    Parameters for Mark to Market estimated value
    训练模型的参数。模型目标是预测中债估值收益率涨跌。
    参数主要是Features和Labels
    """

    ALL_FEATS = [
        # 原始features
        "date",
        "close",
        "fr007",
        "cdb10y",
        "fr0071y",
        "fr0075y",
        "usdcny",
        # 10年国债收益率变种
        "pct_chg",
        "avg_chg_5",
        "avg_chg_10",
        "avg_chg_20",
        "close_avg_5",
        # 流动性指标变种
        "fr007_chg_5",
        "fr0071y_chg_5",
        # 10年国债收益率与其他各种指标之间的差值
        "spread_t1y",
        "spread_cdb10y",
        "spread_fr007",
        "spread_fr0071y",
        "spread_fr0075y",
        "spread_usdcny",
        # 其他各种指标之间的差值
        "spread_fr0075y_fr0071y",
        # 汇率变种
        "usdcny_chg_5",
        # label
        "future",
        "target",
    ]

    TRAIN_FEATS = [
        # 原始features
        "close",
        "fr007",
        # 'cdb10y',
        # 'fr0071y',
        "fr0075y",
        # 'usdcny',
        # 10年国债收益率变种
        "pct_chg",
        "avg_chg_5",
        # 'avg_chg_10',
        "avg_chg_20",
        "close_avg_5",
        # 流动性指标变种
        "fr007_chg_5",
        # 'fr0071y_chg_5',
        # 10年国债收益率与其他各种指标之间的差值
        "spread_t1y",
        # 'spread_cdb10y',
        "spread_fr007",
        # 'spread_fr0071y',
        "spread_fr0075y",
        "spread_usdcny",
        # 其他各种指标之间的差值
        # 'spread_fr0075y_fr0071y',
        # 汇率变种
        "usdcny_chg_5",
    ]
    LABELS = ["target"]


class SPREAD:
    # Bonds must be in ascending order of issue date
    CDB_CODES = [
        "180210",
        "190205",
        "190210",
        "190215",
        "200205",
        "200210",
        "200215",
        "210205",
        "210210",
        "210215",
        "220205",
        "220210",
        "220215",
        "220220",
        "230205",
    ]

    FIXED_FEATS = [
        "MONTH",  # month of the year
        "DAYS_SINCE_LEG2_IPO",  # days since first trading date of leg2
    ]

    #  Features that are calculated according to days_back parameter
    DYNAMIC_FEATS = [
        "SPREAD",  # leg2 ytm - leg1 ytm
        "VOL_DIFF",  # leg2 vol - leg1 vol
        "OUT_BAL_DIFF",  # leg2 outstanding balance - leg1 outstanding balance
    ]


class HistorySimilarity:
    FEATURES = [
        # "t_10y_yield_chg_long",
        "t_10y_yield_chg_short",
        "t_1y_yield_chg_short",
        # "stock_return",
        # "hs300_pctl",
        "stock_return_pctl",
        "t_10y_pctl",
        "t_1y_pctl",
        "t_us_cn_10y_spread_pctl",
        # "t_10y_yield_chg_long_pctl",
        "t_10y_yield_chg_short_pctl",
        "t_1y_yield_chg_short_pctl",
        # "t_10y",
        # "t_1y",
        # "avg_chg_5",
        # "avg_chg_10",
        # "avg_chg_20"
    ]

    # Yield change between 2 dates. yield_chg_fwd_n = t_10y(t+n) - t_10y(t)
    LABELS_YIELD_CHG = {
        5: "yield_chg_fwd_5",
        10: "yield_chg_fwd_10",
        20: "yield_chg_fwd_20",
        30: "yield_chg_fwd_30",
    }

    # Yield change through a window of n days. 3 classes(1, 0 , -1) are assigned according to yield change path.
    # 1: Yield change has been > a HIGH_THRESHOLD for at least one day in the window.
    # 0: Yield change has been between a HIGH_THRESHOLD and a LOW_THRESHOLD for the whole window.
    # -1: Yield change has been < a LOW_THRESHOLD for at least one day in the window.
    LABELS_MULTI_CLASS = {
        5: "multi_class_5",
        10: "multi_class_10",
        20: "multi_class_20",
        30: "multi_class_30",
    }

    PARAMS = {
        "YIELD_PCTL_WINDOW": 5 * 250,
        "YIELD_CHG_PCTL_WINDOW": 5 * 250,
        "YIELD_CHG_WINDOW_LONG": 20,
        "YIELD_CHG_WINDOW_SHORT": 10,
        "STOCK_RETURN_WINDOW": 10,
        "STOCK_RETURN_PCTL_WINDOW": 5 * 250,
        "HS300_PCTL_WINDOW": 5 * 250,
        "HIGH_THRESHOLD": 0.020,  # For multi classification. Unit is %
        "LOW_THRESHOLD": -0.005,  # For multi classification. Unit is %
    }

    SRC_NAME: str = "history_similarity.csv"
    FEATURE_FILE: str = "history_processed.csv"
    SIMI_EUCLIDEAN: str = "similarity_matrix_euclidean.csv"
    SIMI_COSINE: str = "similarity_matrix_cosine.csv"
    PREDICT_FILE: str = "predictions.csv"
