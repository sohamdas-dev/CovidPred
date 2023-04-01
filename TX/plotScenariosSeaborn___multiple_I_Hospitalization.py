import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import pickle

state_name = 'texas'
list_dates_to_label = [92, 184, 276]
##
show_errors_based_on_num_dates_from_prediction = True
num_dates_to_consider = 30
##
inpData = pd.read_csv('./Data/NEW-' + state_name + '-history-Final-Data-with-seasonality-and-sinusoid.csv')
#
first_row_of_parameters = pd.read_csv('./Data/parameter_samples_' + state_name + '.csv').iloc[0].tolist()[1:]
_ ,_, _, _, _, _, _, \
indFirstValidRow, indFirstAvailData, indLastAvailData, indLastValidRow, \
_, _, N0, _, _, _, _ = first_row_of_parameters
indFirstValidRow = int(indFirstValidRow)
indFirstAvailData = int(indFirstAvailData)
indLastAvailData = int(indLastAvailData)
indLastValidRow = int(indLastValidRow)
#
## Loading averall_results and ploting it
overall_result_loaded = pd.read_csv('./Data/overall_results-' + state_name + '.csv')
overall_result_loaded.set_index('t')
overall_result_for_plot = pd.DataFrame({'Hospitalization':overall_result_loaded['Hospitalized_forecast'].tolist()}, index = overall_result_loaded['t'].tolist())
#
overall_true_I_and_hospitalization = pd.read_csv('./Data/overall_true_I-' + state_name + '.csv')
#
true_Hospitalization = pd.DataFrame({'Hospitalization':overall_true_I_and_hospitalization['true_Hospitalization'].tolist()}, index = overall_true_I_and_hospitalization['t'].tolist())
#
forecasted_Hospitalization = overall_result_for_plot.loc[(overall_result_for_plot.index > indLastAvailData)]
#
## plot works: legend inside and outside of the figure box (we will crop the figure)
#
true_Hospitalization_with_label = true_Hospitalization
true_Hospitalization_with_label['source'] = "Real Data"
forecasted_Hospitalization_with_label = forecasted_Hospitalization
forecasted_Hospitalization_with_label['source'] = "Forecasts"
true_and_forecasted_Hospitalization = true_Hospitalization_with_label
true_and_forecasted_Hospitalization = true_and_forecasted_Hospitalization.append(forecasted_Hospitalization, sort=False)
true_and_forecasted_Hospitalization['time'] = true_and_forecasted_Hospitalization.index
#
true_and_forecasted_Hospitalization["Hospitalization"] = true_and_forecasted_Hospitalization["Hospitalization"]*N0
#
plt.figure(1)
sns.lineplot(data=true_and_forecasted_Hospitalization, x="time", y = "Hospitalization", hue="source", ci=99, legend=True)
xtick_labels = [inpData['DateTime'].iloc[i] for i in list_dates_to_label]
plt.xticks(list_dates_to_label, xtick_labels)
plt.legend(loc='upper left')
plt.xlabel('Date', fontsize=14)
plt.rcParams['axes.titley'] = 1.0    # For lowering the location of title
plt.rcParams['axes.titlepad'] = -14  # For lowering the location of title
plt.ylabel('No. of hospitalized individuals', fontsize=14)
# add the dashed line
Hospitalization_val_at_indLastAvailData = true_Hospitalization.iloc[true_Hospitalization.index == indLastAvailData]["Hospitalization"].mean()
plt.plot([indLastAvailData,indLastAvailData],[Hospitalization_val_at_indLastAvailData - 0.000008, Hospitalization_val_at_indLastAvailData + 0.000008], '--k')
#
#
#
## we use the following data later on for calculating the metrics
forecasted_data = true_and_forecasted_Hospitalization[true_and_forecasted_Hospitalization["source"]=="Forecasts"]
mean_of_ci_Hospitalization = forecasted_data.groupby('time')['Hospitalization'].mean()
std_of_ci_Hospitalization = forecasted_data.groupby('time')['Hospitalization'].std()
#
true_data_scaled_back = true_Hospitalization_with_label
true_data_scaled_back["Hospitalization_true"] = true_data_scaled_back["Hospitalization"]*N0
true_data_scaled_back['time'] = true_data_scaled_back.index
true_mean_of_ci_Hospitalization = true_data_scaled_back.groupby('time')['Hospitalization_true'].mean()
#
plots_dataframe = forecasted_Hospitalization
plots_dataframe['time'] = plots_dataframe.index
plots_dataframe["Hospitalization"] = plots_dataframe["Hospitalization"]*N0
#
dict_mean_of_ci_Hospitalization = dict()
for i in true_mean_of_ci_Hospitalization.index:
    dict_mean_of_ci_Hospitalization[i] = true_mean_of_ci_Hospitalization[i]
#
list_Hospitalization_mean_absolute_error = []
list_mean_absolute_percentage_error_Hospitalization = []
list_of_time_in__plots_dataframe = list(plots_dataframe["time"])
list_of_hospitalization_forecasts_in__plots_dataframe = list(plots_dataframe["Hospitalization"])
for i in range(0, len(list_of_time_in__plots_dataframe)):
    curr_time = list_of_time_in__plots_dataframe[i]
    list_Hospitalization_mean_absolute_error.append(  abs(list_of_hospitalization_forecasts_in__plots_dataframe[i] - dict_mean_of_ci_Hospitalization[curr_time])  )
    list_mean_absolute_percentage_error_Hospitalization.append(  100*abs(list_of_hospitalization_forecasts_in__plots_dataframe[i] - dict_mean_of_ci_Hospitalization[curr_time]) / (abs(dict_mean_of_ci_Hospitalization[curr_time]) + 0.0000000000001)  )
#
plots_dataframe["Hospitalization_mean_absolute_error"] = list_Hospitalization_mean_absolute_error
plots_dataframe["mean_absolute_percentage_error_Hospitalization"] = list_mean_absolute_percentage_error_Hospitalization
#
#
#
#
#
#
#
#
#
## ## calculating the evaluation metric
if show_errors_based_on_num_dates_from_prediction:
    plots_dataframe_trimmed = plots_dataframe[plots_dataframe["time"] < min(plots_dataframe["time"]) + num_dates_to_consider]
    plots_dataframe_trimmed["time_trimmed"] = plots_dataframe_trimmed["time"] - min(plots_dataframe["time"])
    #
    plt.figure(2)
    curr_relplot = sns.lineplot(data=plots_dataframe_trimmed, x="time_trimmed", y = "Hospitalization_mean_absolute_error", hue="source", ci='sd', legend=False)
    # xtick_labels = [inpData['DateTime'].iloc[i] for i in list_dates_to_label]
    # plt.xticks(list_dates_to_label, xtick_labels)
    plt.legend(loc='upper left')
    plt.xlabel('Time (Day)', fontsize=14)
    plt.rcParams['axes.titley'] = 1.0    # For lowering the location of title
    plt.rcParams['axes.titlepad'] = -14  # For lowering the location of title
    plt.ylabel('MAE of hospitalized individuals', fontsize=14) # mean absolute error
    #
    #
    plt.figure(3)
    curr_relplot = sns.lineplot(data=plots_dataframe_trimmed, x="time_trimmed", y = "mean_absolute_percentage_error_Hospitalization", hue="source", ci='sd', legend=False)
    # xtick_labels = [inpData['DateTime'].iloc[i] for i in list_dates_to_label]
    # plt.xticks(list_dates_to_label, xtick_labels)
    plt.legend(loc='upper left')
    plt.xlabel('Time (Day)', fontsize=14)
    plt.rcParams['axes.titley'] = 1.0    # For lowering the location of title
    plt.rcParams['axes.titlepad'] = -14  # For lowering the location of title
    plt.ylabel('MAPE of hospitalized individuals (%)', fontsize=14) # mean absolute percentage error
else:
    plt.figure(2)
    curr_relplot = sns.lineplot(data=plots_dataframe, x="time", y="Hospitalization_mean_absolute_error", hue="source",ci='sd', legend=False)
    xtick_labels = [inpData['DateTime'].iloc[i] for i in list_dates_to_label]
    plt.xticks(list_dates_to_label, xtick_labels)
    plt.legend(loc='upper left')
    plt.xlabel('Time (Day)', fontsize=14)
    plt.rcParams['axes.titley'] = 1.0  # For lowering the location of title
    plt.rcParams['axes.titlepad'] = -14  # For lowering the location of title
    plt.ylabel('MAE of hospitalized individuals', fontsize=14)  # mean absolute error
    #
    #
    plt.figure(3)
    curr_relplot = sns.lineplot(data=plots_dataframe, x="time", y="mean_absolute_percentage_error_Hospitalization",hue="source", ci='sd', legend=False)
    xtick_labels = [inpData['DateTime'].iloc[i] for i in list_dates_to_label]
    plt.xticks(list_dates_to_label, xtick_labels)
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)', fontsize=14)
    plt.rcParams['axes.titley'] = 1.0  # For lowering the location of title
    plt.rcParams['axes.titlepad'] = -14  # For lowering the location of title
    plt.ylabel('MAPE of hospitalized individuals (%)', fontsize=14)  # mean absolute percentage error
#
#
## ## Saving the figure
curr_ax = plt.gcf()
fig_name = state_name + '_' + "Hospitalization" + '_' + str(indFirstAvailData) + '_' + str(indLastAvailData) + '_' + str(
    indFirstValidRow) + '_' + str(indLastValidRow)
with open('savedFigs/' + fig_name, 'wb') as f:
    pickle.dump(curr_ax, f)
plt.close()
hhh = 5+6
#
5 + 6