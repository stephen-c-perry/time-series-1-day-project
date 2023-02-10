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




def holt(train, val, test):
    yhat_df = val + train.diff(365).mean()
    yhat_df.index = test.index

    for col in train.columns:
        model = Holt(train[col], exponential = False)
        model = model.fit(smoothing_level = .1, 
                          smoothing_slope = .1, 
                          optimized = False)
        yhat_items = model.predict(start = val.index[0], 
                                   end = val.index[-1])
        yhat_df[col] = round(yhat_items, 2)


def plot_and_eval(train, val, test, yhat_df, target_var):
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(val[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = round(sqrt(mean_squared_error(val['avg_temp'], yhat_df['avg_temp'])), 4)
    print(target_var, '-- RMSE: {:.4f}'.format(rmse))
    plt.show()




def plot_all_target_sets(train, val, test):
    for col in train.columns:

        plt.figure(figsize=(12,4))
        plt.plot(train[col])
        plt.plot(val[col])
        plt.plot(test[col])
        plt.ylabel(col)
        plt.title(col)
        plt.show()



def plot_and_evaluate(train):
    for col in train.columns:
        plot_and_eval(target_var = col)


