import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_monthly_avg(train):
    train.resample('Y').mean().plot(title= 'Monthly Average Temp Increases Over Time')