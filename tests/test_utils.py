import pytest
import joblib
# import tensorflow.keras
import xgboost as xgb
from fxincome.utils import ModelAttr, JsonModel, cal_coupon
from fxincome.const import PATH
import datetime


class TestModel:
    @pytest.fixture(scope='class')
    def global_data(self):
        lstm_name = 'Checkpoint-10-SEQ-1-PRED-20210903-1639.model'
        lstm_features = [
            'close',  # scale
            'amount',  # scale
            't10y',  # scale
            'fr007_5y',  # scale
            'pct_chg',
            'avg_chg_10',
            'spread_t10y',
            'spread_fr007',
            'spread_fr007_5y',
            'spread_usdcny'  # scale
        ]
        lstm_scaled_feats = ['close', 'amount', 't10y', 'fr007_5y', 'spread_usdcny']
        lstm_stats = joblib.load(PATH.YTM_MODEL + 'stats-10-SEQ-1-PRED-20210903-1639.pkl')
        lstm_labels = {'target': {'value_scope': '[0,1]'}}
        lstm_model = ModelAttr(
            name=lstm_name,
            features=lstm_features,
            labels=lstm_labels,
            scaled_feats=lstm_scaled_feats,
            stats=lstm_stats
        )
        xgb_name = 'spread_0.592_XGB_20230419_0907.json'
        xgb_features = ['close', 'pct_chg', 'avg_chg_5', 'avg_chg_10', 'fr007_chg_5', 'spread_t1y',
                        'spread_fr007', 'spread_usdcny', 'usdcny_chg_5']
        xgb_labels = {'LABEL': {'value_scope': '[0,1]',
                                'days_forward': 10,
                                'spread_threshold': -0.01}}
        xgb_other = {'days_back': 5, 'n_samples': 190, 'last_n_bonds_for_test': 3,
                     'bonds': ["180210", "190205",
                               "190210", "190215",
                               "200205", "200210",
                               "200215", "210205",
                               "210210", "210215",
                               "220205", "220210",
                               "220215", "220220"]}
        xgb_model = ModelAttr(
            name=xgb_name,
            features=xgb_features,
            labels=xgb_labels,
            other=xgb_other
        )

        lr_name = 'spread_0.627_LR_20230420_1134.pkl'
        lr_model = ModelAttr(
            name=lr_name,
            features=xgb_features,
            labels=xgb_labels,
            other=xgb_other
        )
        return {
            'lstm_name': lstm_name,
            'lstm_features': lstm_features,
            'lstm_labels': lstm_labels,
            'lstm_scaled_feats': lstm_scaled_feats,
            'lstm_stats': lstm_stats,
            'xgb_name': xgb_name,
            'xgb_features': xgb_features,
            'xgb_labels': xgb_labels,
            'xgb_other': xgb_other,
            'lr_name': lr_name,
            'lr_features': xgb_features,
            'lr_labels': xgb_labels,
            'lstm_model': lstm_model,
            'xgb_model': xgb_model,
            'lr_model': lr_model
        }

    def test_save_attr(self, global_data):
        lstm_model = global_data['lstm_model']
        JsonModel.save_attr(lstm_model, PATH.YTM_MODEL + JsonModel.JSON_NAME)
        model = JsonModel.load_attr(lstm_model.name, PATH.YTM_MODEL + JsonModel.JSON_NAME)
        assert lstm_model.name == model.name
        assert lstm_model.features == model.features
        assert lstm_model.labels == model.labels
        assert lstm_model.scaled_feats == model.scaled_feats
        assert lstm_model.stats == model.stats
        JsonModel.save_attr(global_data['xgb_model'], PATH.SPREAD_MODEL + JsonModel.JSON_NAME)
        JsonModel.save_attr(global_data['lr_model'], PATH.SPREAD_MODEL + JsonModel.JSON_NAME)

    def test_load_attr(self, global_data):
        xgb_model = global_data['xgb_model']
        model = JsonModel.load_attr(xgb_model.name, PATH.SPREAD_MODEL + JsonModel.JSON_NAME)
        assert xgb_model.name == model.name
        assert xgb_model.features == model.features
        assert xgb_model.labels == model.labels
        assert JsonModel.load_attr('Non-Exists', PATH.SPREAD_MODEL + JsonModel.JSON_NAME) is None

    def test_delete_attr(self, global_data):
        xgb_model = global_data['xgb_model']
        JsonModel.delete_attr(xgb_model.name, PATH.SPREAD_MODEL + JsonModel.JSON_NAME)
        assert JsonModel.load_attr(xgb_model.name, PATH.SPREAD_MODEL + JsonModel.JSON_NAME) is None
        JsonModel.save_attr(global_data['xgb_model'], PATH.SPREAD_MODEL + JsonModel.JSON_NAME)
        model = JsonModel.load_attr(xgb_model.name, PATH.SPREAD_MODEL + JsonModel.JSON_NAME)
        assert model.labels == xgb_model.labels

    def test_load_plain_models(self, global_data):
        plain_names = [global_data['xgb_name']]
        plain_dict = JsonModel.load_plain_models(plain_names, PATH.SPREAD_MODEL, 'xgb')
        xgb_model = xgb.Booster()
        xgb_model.load_model(PATH.SPREAD_MODEL + global_data['xgb_name'])
        clf = xgb.XGBClassifier()
        clf._Booster = xgb_model
        plain_model = plain_dict[global_data['xgb_model']]
        assert plain_model.get_params()['gamma'] == clf.get_params()['gamma']
        plain_names = [global_data['lr_name']]
        plain_dict = JsonModel.load_plain_models(plain_names, PATH.SPREAD_MODEL, 'joblib')
        lr_model = joblib.load(PATH.SPREAD_MODEL + global_data['lr_name'])
        plain_model = plain_dict[global_data['lr_model']]
        assert plain_model.get_params()['logistic__C'] == lr_model.get_params()['logistic__C']


def test_cal_coupon():
    """Test coupon calculation for bond 240004.IB"""
    # Bond parameters
    issue_date = datetime.date(2024, 2, 25)
    maturity_date = datetime.date(2034, 2, 25)
    coupon = 0.0235
    coupon_freq = 2
    
    # Check period
    chk_start = datetime.date(2025, 2, 24)
    chk_end = datetime.date(2025, 2, 26)
    
    expected_coupon = 0.01175
    
    actual_coupon = cal_coupon(
        chk_start=chk_start,
        chk_end=chk_end,
        issue_date=issue_date,
        maturity_date=maturity_date,
        coupon=coupon,
        coupon_freq=coupon_freq
    )
    
    # Assert that calculated value matches expected value
    assert pytest.approx(actual_coupon, abs=1e-6) == expected_coupon, f"Expected coupon {expected_coupon}, got {actual_coupon}"

    # Check period
    chk_start = datetime.date(2025, 2, 24)
    chk_end = datetime.date(2025, 2, 25) # End date is included.
    
    expected_coupon = 0.01175  # End date is the payment date and included in the checking period.

    actual_coupon = cal_coupon(
        chk_start=chk_start,
        chk_end=chk_end,
        issue_date=issue_date,
        maturity_date=maturity_date,
        coupon=coupon,
        coupon_freq=coupon_freq
    )
    assert pytest.approx(actual_coupon, abs=1e-6) == expected_coupon, f"Expected coupon {expected_coupon}, got {actual_coupon}"
