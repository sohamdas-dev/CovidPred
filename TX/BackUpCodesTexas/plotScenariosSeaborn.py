import pandas as pd
import numpy as np
from DataPrep import dataPreparation
from FittingModel import fittingModel
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import pickle

state_name = 'texas'
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
## Loading averall_results and ploting it
overall_result_loaded = pd.read_csv('./Data/overall_results-' + state_name + '.csv')
overall_result_loaded.set_index('t')
overall_result_for_plot = pd.DataFrame({'I':overall_result_loaded['I'].tolist()}, index = overall_result_loaded['t'].tolist())
true_I_data = pd.read_csv('./Data/NEW-' + state_name + '-history-Final-Data-with-seasonality-and-sinusoid.csv')
#
sns.relplot(data=overall_result_for_plot, kind="line")
plt.plot(list(range(indFirstValidRow, indLastValidRow)), true_I_data['I'].iloc[indFirstValidRow:indLastValidRow] / N0, 'r', linewidth=2, label='True I')
plt.plot([indFirstAvailData, indFirstAvailData], [0, true_I_data['I'].iloc[indFirstAvailData]/N0], '--k')
plt.plot([indLastAvailData, indLastAvailData], [0, true_I_data['I'].iloc[indLastAvailData]/N0], '--k')
llll = [50, 100, 150, 200, 250]
xtick_labels = [str(i) + "::" + inpData['DateTime'].iloc[indFirstValidRow + i] for i in llll]
plt.xticks(llll, xtick_labels)
plt.title(state_name)
# curr_ax = plt.gca()
curr_ax = plt.gcf()
#
fig_name = state_name + '_' + str(indFirstAvailData) + '_' + str(indLastAvailData) + '_' + str(indFirstValidRow) + '_' + str(indLastValidRow)
with open('savedFigs/' + fig_name, 'wb') as f:
    pickle.dump(curr_ax, f)
plt.show()
plt.close()
zzz = 5 + 6
loaded_ax = pickle.load(open('savedFigs/' + fig_name, "rb"))
#
plt.xlabel('t', fontsize=16)
plt.ylabel('I', fontsize=16)
loaded_ax.show()
5 + 6