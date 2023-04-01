import pandas as pd
import numpy as np
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
#
## Loading averall_results and ploting it
overall_result_loaded = pd.read_csv('./Data/overall_results-' + state_name + '.csv')
overall_result_loaded['A_fr_10_minus_A_fr_00'] = overall_result_loaded['A_fr_10'] - overall_result_loaded['A_fr_00']
overall_result_loaded['A_fr_11_minus_A_fr_01'] = overall_result_loaded['A_fr_11'] - overall_result_loaded['A_fr_01']
overall_result_loaded['ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01'] = overall_result_loaded['A_fr_10_minus_A_fr_00'] / overall_result_loaded['A_fr_11_minus_A_fr_01']
#
overall_result_loaded['A_fd_00_minus_A_fd_10'] = overall_result_loaded['A_fd_00'] - overall_result_loaded['A_fd_10']
overall_result_loaded['A_fd_01_minus_A_fd_11'] = overall_result_loaded['A_fd_01'] - overall_result_loaded['A_fd_11']
overall_result_loaded['ratio___A_fd_00_minus_A_fd_10___to___A_fd_01_minus_A_fd_11'] = overall_result_loaded['A_fd_00_minus_A_fd_10'] / overall_result_loaded['A_fd_01_minus_A_fd_11']
overall_result_loaded.set_index('t')
#
conditions = [
    (overall_result_loaded['A_fd_00_minus_A_fd_10'] >= 0) & (overall_result_loaded['A_fd_01_minus_A_fd_11'] >= 0),
    (overall_result_loaded['A_fd_00_minus_A_fd_10'] >= 0) & (overall_result_loaded['A_fd_01_minus_A_fd_11'] < 0),
    (overall_result_loaded['A_fd_00_minus_A_fd_10'] < 0) & (overall_result_loaded['A_fd_01_minus_A_fd_11'] < 0),
    (overall_result_loaded['A_fd_00_minus_A_fd_10'] < 0) & (overall_result_loaded['A_fd_01_minus_A_fd_11'] >= 0)]
choices = ['Q1', 'Q2', 'Q3', 'Q4']
overall_result_loaded['Q'] = np.select(conditions, choices)
#
leng = max(overall_result_loaded['ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01']) - min(overall_result_loaded['ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01'])
if False:
    plt.text(x=max(overall_result_loaded['ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01']) - 0.05 * leng,
             y=max(overall_result_loaded['ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01']) - 0.1 * leng,
             s=r"$\measuredangle 45^{\circ}$",
             # fontdict=dict(color="red",size=10),
             # bbox=dict(facecolor="yellow",alpha=0.5)
             )
#
if False:
    sns.scatterplot(data=overall_result_loaded, x='ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01', y='ratio___A_fd_00_minus_A_fd_10___to___A_fd_01_minus_A_fd_11', legend=True, hue='Q')
else:
    sns.scatterplot(data=overall_result_loaded, x='ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01',y='ratio___A_fd_00_minus_A_fd_10___to___A_fd_01_minus_A_fd_11')
sns.lineplot(data=overall_result_loaded, x='ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01', y='ratio___A_fr_10_minus_A_fr_00___to___A_fr_11_minus_A_fr_01', legend=False, color='red',linewidth=1, linestyle='--')
#
plt.xlabel(r"$\frac{T_1 - R_1}{P_1 - S_1}$", fontsize=14)
plt.ylabel(r"$\frac{R_0 - T_0}{S_0 - P_0}$", fontsize=14)
#
# plt.savefig("foo.pdf", bbox_inches='tight')
plt.savefig("foo.pdf")
zzz = 5 + 6