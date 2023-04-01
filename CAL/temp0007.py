import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import pickle

state_name = 'california'
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
# plt.figure(figsize=(4,4))
# curr_ax = plt.axes()
# curr_ax.annotate('$t$', xy=(0.98, 0), ha='left', va='top', xycoords='axes fraction', fontsize=14)
# curr_ax.annotate('$x$', xy=(0, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=14)
true_I_with_label = true_I
true_I_with_label['source'] = "True_I"
forecasted_I_with_label = forecasted_I
forecasted_I_with_label['source'] = "Forecasted_I"
true_and_forecasted_I = true_I_with_label
true_and_forecasted_I = true_and_forecasted_I.append(forecasted_I, sort=False)
true_and_forecasted_I['time'] = true_and_forecasted_I.index
# curr_relplot = sns.relplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", kind="line", palette=["dimgray", "r"], legend=True, ci=0.99)
# markers = {"True_I": "s", "Forecasted_I": "X"}
# curr_relplot = sns.scatterplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", palette=["dimgray", "r"], legend=True, markers=markers)
curr_relplot = sns.relplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", kind="line", palette=["dimgray", "r"], legend=True, ci='sd')
# curr_relplot = sns.relplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", palette=["dimgray", "r"], legend=True, ci=0.99, markers=markers, style='source')
# curr_relplot = sns.relplot(data=true_and_forecasted_I, x="time", y = "I", s=10, hue="source", palette=["blue", "red"], legend=True)
# curr_relplot = sns.lineplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", palette=["dimgray", "r"], legend=True, ci=0.6)
#
#
# llll = [50, 100, 150, 200, 250]
llll = [50, 150, 250]
# xtick_labels = [str(i) + "::" + inpData['DateTime'].iloc[indFirstValidRow + i] for i in llll]
xtick_labels = [inpData['DateTime'].iloc[indFirstValidRow + i] for i in llll]
plt.xticks(llll, xtick_labels)
plt.legend(loc='upper left')
plt.xlabel('t', fontsize=14)
plt.ylabel('I - ' + state_name.capitalize(), fontsize=14)
# add the dashed line
I_val_at_indLastAvailData = true_I.iloc[true_I.index == indLastAvailData]["I"].mean()
plt.plot([indLastAvailData,indLastAvailData],[I_val_at_indLastAvailData - 0.001, I_val_at_indLastAvailData + 0.001], '--k')
#
curr_ax = plt.gcf()
fig_name = state_name + '_' + str(indFirstAvailData) + '_' + str(indLastAvailData) + '_' + str(
    indFirstValidRow) + '_' + str(indLastValidRow)
with open('savedFigs/' + fig_name, 'wb') as f:
    pickle.dump(curr_ax, f)
plt.close()
hhh = 5+6
#
5 + 6