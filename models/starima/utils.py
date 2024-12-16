import pandas as pd
from typing import Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_and_predict_sarima(train_data: pd.DataFrame, steps) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    
    for cell in train_data.columns:
        ts = train_data[cell]
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts.index = pd.to_datetime(ts.index)

        if ts.sum() == 0:
            predictions.append([0] * steps)
            
            continue

        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps)
        forecast.index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=steps)
        
        predictions.append(forecast)
    
    predictions_df = pd.DataFrame(predictions).T
    predictions_df.columns = train_data.columns

    return predictions_df
