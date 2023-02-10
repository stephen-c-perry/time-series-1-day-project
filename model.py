import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wrangle as w
import explore as e

from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt




def X_y_modeling_split(train, val, test, target_col):
    y_train = train[[target_col]]
    X_train = train.drop(columns=[target_col])
    
    y_val = val[[target_col]]
    X_val = val.drop(columns = [target_col])
    
    y_test = test[[target_col]]
    X_test = test.drop(columns = [target_col])

    return X_train, y_train, X_val, y_val, X_test, y_test




def rolling_avg_model(train, val):
    period = 365

    avg_temp = round(train['avg_temp'].rolling(period).mean().iloc[-1], 4)

    # yhat_df = make_predictions()

    yhat_df = pd.DataFrame({'avg_temp': [avg_temp]}, index = val.index)
    yhat_df.head()

    return yhat_df




def plot_and_eval(target_var):
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

