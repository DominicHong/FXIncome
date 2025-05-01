import json, joblib
import xgboost as xgb
import datetime
from typing import Literal, Union
from dataclasses import dataclass, field
from financepy.products.bonds import Bond
from financepy.utils import DayCountTypes, FrequencyTypes, Date


@dataclass(frozen=True)
class ModelAttr:
    """
    A class that stores the attributes of a trained model. It can be serialized in a json format.
        Args:
            name(str): 模型的名字，唯一能hash的field，是ModelAttr的唯一标识
            features(list): 模型的特征，字符串list
            labels(dict): 模型的预测目标，数据结构为Dict of Dict
                          {label1: {value_scope: str, days_forward: int, threshold: float},
                          label2: ...}
            scaled_features(list): 需要做scaling的特征，可为空
            stats(dict): 训练集做Scaling的统计特征，数据结构为Dict of Dict
                         对于zscore， {feature1: {mean: float, std: float}, feature2: ...}
                         对于minmax， {feature1: {min: float, max: float}, feature2: ...}
            other(dict): 其他信息，可为空
                         {  #  以下定义详见： spread.train_model.generate_dataset()
                         days_back: int,
                         n_samples: int,
                         last_n_bonds_for_test: int,
                         bonds: List[str]
                         }
    """

    name: str
    features: list = field(compare=False)
    labels: dict = field(compare=False)
    scaled_feats: list = field(default=None, compare=False)
    stats: dict = field(default=None, compare=False)
    other: dict = field(default=None, compare=False)


class JsonModel:
    """
    A database class that CRUD ModelAttrs in a single Json file. Users can query and save model attributes by this class.
    All the ModelAttrs are stored in a single Json file.
    The json file is located at json_path. The file name is model_attrs.json.
    """

    JSON_NAME = "model_attrs.json"

    @staticmethod
    def save_attr(model_attr: ModelAttr, json_file: str) -> None:
        """
        将某个模型对应的要素添加至本地文件，原来的模型要素不会改变。
            Args:
                model_attr(ModelAttr): 将要存储的Model要素的对象
                json_file(str): Json路径及文件名
        """
        name = model_attr.name
        #  Save model to json file. New model is added to the existing models.
        try:
            with open(json_file, "r") as f:
                model_dict = json.load(f)
        except FileNotFoundError:
            model_dict = {}
        model_dict[name] = model_attr.__dict__
        with open(json_file, "w") as f:
            json.dump(model_dict, f)

    @staticmethod
    def load_attr(name: str, json_file: str) -> Union[ModelAttr, None]:
        """
        从本地读取某个模型的要素
            Args:
                name(str): 模型名字
                json_file(str): Json路径及文件名
            Returns:
                model_attr(ModelAttr): 以ModelAttr形式返回Model的要素，如没有这个名字的模型，则返回None
        """
        try:
            with open(json_file, "r") as f:
                model_dict = json.load(f)
            model = ModelAttr(
                name,
                model_dict[name]["features"],
                model_dict[name]["labels"],
                model_dict[name]["scaled_feats"],
                model_dict[name]["stats"],
                model_dict[name]["other"],
            )
        except (FileNotFoundError, KeyError):
            return None
        return model

    @staticmethod
    def delete_attr(name: str, json_file: str) -> None:
        """
        从本地文件中删除某个模型的要素
            Args:
                name(str): 模型名字
                json_file(str): Json全路径及文件名
        """
        try:
            with open(json_file, "r") as f:
                model_dict = json.load(f)
        except FileNotFoundError:
            return None
        model_dict.pop(name)
        with open(json_file, "w") as f:
            json.dump(model_dict, f)

    @staticmethod
    def load_plain_models(
        names: list, json_model_path: str, serialization_type: Literal["joblib", "xgb"]
    ) -> dict:
        """
        从本地读取某个传统模型的要素和模型。返回的plain models都是符合sklearn接口的Classifier。
            Args:
                names(list): a list of strs. 传统模型名字列表
                json_model_path(str): Json文件和模型文件的路径，Json和模型必须在同一个文件夹下。
                serialization_type(str): 模型序列化时使用的方法，可选'joblib', 'xgb'
            Returns:
                plain_dict(dict): key: ModelAttr, value: model
        """
        attrs = [
            JsonModel.load_attr(name, json_model_path + JsonModel.JSON_NAME)
            for name in names
        ]
        plain_dict = {}
        for attr in attrs:
            if serialization_type == "joblib":
                plain_dict[attr] = joblib.load(json_model_path + attr.name)
            elif serialization_type == "xgb":
                xgb_model = xgb.Booster()
                xgb_model.load_model(json_model_path + attr.name)
                clf = xgb.XGBClassifier()
                clf._Booster = xgb_model
                plain_dict[attr] = clf
            else:
                raise TypeError("Invalid serialization type.")
        return plain_dict


def cal_coupon(
    chk_start: datetime.date,
    chk_end: datetime.date,
    issue_date: datetime.date,
    maturity_date: datetime.date,
    coupon: float,
    coupon_freq: int,
) -> float:
    """
    Calculate the coupon payments (if any) during a specified period for a bond. 
    Start date and end date are trade days. 
    One or multiple coupon payments may be between them.
    Bond's tenor must be longer than 1 year.
    
    Args:
        chk_start(datetime.date): The start date of checking period for coupon payment.
        chk_end(datetime.date): The end date of checking period for coupon payment.
        issue_date(datetime.date): The issue date of the bond.
        maturity_date(datetime.date): The maturity date of the bond.
        coupon(float): Coupon rate, 0.025 -> 2.5%
        coupon_freq(int): The number of coupon payments per year. 1 for annual, 2 for semi-annual, 4 for quarterly.
    Returns:
        c(float): The coupon payment per 1 face value.
    """
    if coupon_freq == 1:
        freq_type = FrequencyTypes.ANNUAL
    elif coupon_freq == 2:
        freq_type = FrequencyTypes.SEMI_ANNUAL
    elif coupon_freq == 4:
        freq_type = FrequencyTypes.QUARTERLY
    else:
        raise ValueError("Invalid coupon frequency.")

    chk_start = Date(chk_start.day, chk_start.month, chk_start.year)
    chk_end = Date(chk_end.day, chk_end.month, chk_end.year)
    issue_date = Date(issue_date.day, issue_date.month, issue_date.year)
    maturity_date = Date(maturity_date.day, maturity_date.month, maturity_date.year)
    accrual_type = DayCountTypes.ACT_ACT_ICMA
    bond = Bond(issue_date, maturity_date, coupon, freq_type, accrual_type)

    c = 0
    # The first date is the issue date, which does not have coupon payment.
    for dt, cf in zip(bond.cpn_dts, bond.flow_amounts):
        if chk_start <= dt <= chk_end:
            c += cf
    return c


