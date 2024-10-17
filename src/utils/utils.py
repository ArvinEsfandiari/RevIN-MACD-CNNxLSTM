from collections.abc import MutableMapping
import pickle
from hashlib import md5
import pandas as pd

def TFConvertor(data, newtf):
    # refining data and fill missed values with nan to have a good resampling
    # find frequency of data (minimum freq)
    frequncies = (pd.Series(data.index[1:]) -
                  pd.Series(data.index[:-1])).value_counts()
    # generate time series in data period with its freq
    fine_indices = pd.period_range(
        data.index.min(), data.index.max(), freq=frequncies.index.min())
    fine_indices = fine_indices.to_timestamp()
    # refine indices (empty ones fill with nan)
    data = data.reindex(fine_indices)
    #  تا اینجا میاد میگه که هرجا ایرادی توی تایم فریم بود بیا درستش کن و اونو با نَن پر کن. پس الان یه دیتا داریم که تایم فریمش تمیزه



    # resampling  // first,last,min,max,mean,sum,...  //T or M for min, H for hour, D for day, M, Y,...
    Open = data['open'].resample(newtf).first()
    Open = Open.to_frame()
    Close = data['close'].resample(newtf).last()
    Close = Close.to_frame()
    High = data['high'].resample(newtf).max()
    High = High.to_frame()
    Low = data['low'].resample(newtf).min()
    Low = Low.to_frame()
    Volume = data['tick_volume'].resample(newtf).sum()
    Volume = Volume.to_frame()

    newtfdata = Open
    newtfdata['high'] = High['high']
    newtfdata['low'] = Low['low']
    newtfdata['close'] = Close['close']
    newtfdata['tick_volume'] = Volume['tick_volume']

    # remove nan vals
    newtfdata = newtfdata.dropna()

    return newtfdata

def CreateTimeFrames(data, timeframes):
    new_data = {}
    for t in timeframes:
        new_data[t] = TFConvertor(data, t)
    return new_data