from WindPy import w
import pandas as pd
from dateutil.utils import today
from sqlalchemy import create_engine
from datetime import datetime
from datetime import timedelta
from fxincome import const



def wind_strategies_pool_history_similarity(engine):
    begin=pd.read_sql("select max(date) from strategies_pool_history_similarity",engine).iat[0,0]
    begin=begin.strftime("%Y-%m-%d")
    end=(datetime.today()+timedelta(-1)).strftime("%Y-%m-%d")
    wind_data=w.edb("S0059744,S0059749,G0000886,G0000891,000300.SH", begin, end)
    if len(wind_data.Times)==1:
        return
    df=pd.DataFrame([wind_data.Times]+wind_data.Data,index=['date','t_1y','t_10y','t_us_1y','t_us_10y','hs300']).T
    df[1:].to_sql('strategies_pool_history_similarity',engine,if_exists='append',index=False)
def wind_strategies_pool_511260sh(engine):
    begin=pd.read_sql("select max(date) from strategies_pool_511260sh",engine).iat[0,0]
    begin=begin.strftime("%Y-%m-%d")
    end=(datetime.today()+timedelta(-1)).strftime("%Y-%m-%d")
    wind_data1=w.wsd("511260.SH", "open,high,low,close,volume,turn", begin, end)
    if len(wind_data1.Times)==1:
        return
    df1=pd.DataFrame([wind_data1.Times]+wind_data1.Data,index=['date','open','high','low','close','volume','turnover']).T
    wind_data2=w.wsd("204001.SH", "vwap,close", begin, end)
    df2 = pd.DataFrame([wind_data2.Times] + wind_data2.Data,index=['date','GC001_avg', 'GC001_close']).T
    df=pd.merge(df1,df2,on='date')
    df[1:].to_sql('strategies_pool_511260sh',engine,if_exists='append',index=False)
def main():
    connection_string = (f'mysql+mysqlconnector://{const.MYSQL_CONFIG.USER}:{const.MYSQL_CONFIG.PASSWORD}'
                         f'@{const.MYSQL_CONFIG.HOST}/{const.MYSQL_CONFIG.DATABASE}')
    engine = create_engine(connection_string)
    w.start()


    wind_strategies_pool_history_similarity(engine)
    wind_strategies_pool_511260sh(engine)


    w.close()
    engine.dispose()
    pass
if __name__=='__main__':
    main()