import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import pickle

state_name = 'texas'
list_dates_to_label = [92, 184, 276]
##
# READ the inout Data
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
## READ results
overall_result_loaded = pd.read_csv('./Data/overall_results-' + state_name + '.csv')
overall_result_loaded.set_index('t')
overall_result_for_plot = pd.DataFrame({'I':overall_result_loaded['I'].tolist()}, index = overall_result_loaded['t'].tolist())
#
overall_true_I = pd.read_csv('./Data/overall_true_I-' + state_name + '.csv')
#
true_I = pd.DataFrame({'I':overall_true_I['true_I'].tolist()}, index = overall_true_I['t'].tolist())
#
forecasted_I = overall_result_for_plot.loc[(overall_result_for_plot.index > indLastAvailData)]
#
## plot works: legend inside and outside of the figure box (we will crop the figure)
#
true_I_with_label = true_I
true_I_with_label['source'] = "Real Data"
forecasted_I_with_label = forecasted_I
forecasted_I_with_label['source'] = "Forecasts"
true_and_forecasted_I = true_I_with_label
true_and_forecasted_I = true_and_forecasted_I.append(forecasted_I, sort=False)
true_and_forecasted_I['time'] = true_and_forecasted_I.index
#
true_and_forecasted_I["I"] = true_and_forecasted_I["I"] * N0
# PLOT
sns.lineplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", ci=99, legend=False)
curr_relplot = sns.lineplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", ci='sd')
#
xtick_labels = [inpData['DateTime'].iloc[i] for i in list_dates_to_label]
plt.xticks(list_dates_to_label, xtick_labels)
plt.legend(loc='upper left')
plt.rcParams['axes.titley'] = 1.0    # For lowering the location of title
plt.rcParams['axes.titlepad'] = -14  # For lowering the location of title
# plt.xlabel('Date (YYYY-MM-DD)', fontsize=14)
plt.xlabel('Date', fontsize=14)
# plt.ylabel('I', fontsize=14)
plt.ylabel("No. of infectious individuals", fontsize=14)
# plt.title(state_name.capitalize(), fontsize=14)
## add the dashed line
I_val_at_indLastAvailData = true_I.iloc[true_I.index == indLastAvailData]["I"].mean()
plt.plot([indLastAvailData,indLastAvailData],[(I_val_at_indLastAvailData - 0.0032)*N0, (I_val_at_indLastAvailData + 0.0032)*N0], '--k')
#
curr_ax = plt.gcf()
fig_name = state_name + '_' + str(indFirstAvailData) + '_' + str(indLastAvailData) + '_' + str(
    indFirstValidRow) + '_' + str(indLastValidRow)
with open('savedFigs/' + fig_name, 'wb') as f:
    pickle.dump(curr_ax, f)
curr_ax.savefig('savedFigs/'+fig_name+'.png') # save as png as well
plt.close()
hhh = 5+6
#
5 + 6