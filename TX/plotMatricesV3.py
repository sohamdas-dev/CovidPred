import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# from pylab import *
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
overall_result_loaded = pd.read_csv('./Data/Afd_Afr_matrices-' + state_name + '.csv')
overall_result_loaded.set_index('indLastAvailData')
#
"""
overall_result_for_plot =                                     pd.DataFrame({'A':overall_result_loaded['A_fd_00'].tolist(), "Payoff Matrix Entry":"R0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fd_01'].tolist(), "Payoff Matrix Entry":"S0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fd_10'].tolist(), "Payoff Matrix Entry":"T0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fd_11'].tolist(), "Payoff Matrix Entry":"P0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_00'].tolist(), "Payoff Matrix Entry":"R1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_00'].tolist(), "Payoff Matrix Entry":"S1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_00'].tolist(), "Payoff Matrix Entry":"T1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_00'].tolist(), "Payoff Matrix Entry":"P1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
"""
#
overall_result_for_plot =                                     pd.DataFrame({'A':overall_result_loaded['A_fd_00'].tolist(), "Payoff Matrix Entry":"R0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fd_01'].tolist(), "Payoff Matrix Entry":"S0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fd_10'].tolist(), "Payoff Matrix Entry":"T0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fd_11'].tolist(), "Payoff Matrix Entry":"P0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_00'].tolist(), "Payoff Matrix Entry":"R1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_01'].tolist(), "Payoff Matrix Entry":"S1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
# overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_10'].tolist(), "Payoff Matrix Entry":"T1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
overall_result_for_plot = pd.concat([overall_result_for_plot, pd.DataFrame({'A':overall_result_loaded['A_fr_11'].tolist(), "Payoff Matrix Entry":"P1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
#
overall_result_for_plot_ratios =                                            pd.DataFrame({'ratio':(overall_result_loaded['A_fd_00']/overall_result_loaded['A_fd_11']).tolist(), "Ratio": "R0/P0", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())
overall_result_for_plot_ratios = pd.concat([overall_result_for_plot_ratios, pd.DataFrame({'ratio':(overall_result_loaded['A_fr_00']/overall_result_loaded['A_fr_11']).tolist(), "Ratio": "R1/P1", "indLastAvailData":overall_result_loaded['indLastAvailData'].tolist(), "DateLastAvailData":inpData['DateTime'].iloc[(indFirstValidRow + overall_result_loaded['indLastAvailData'])].tolist()}, index = overall_result_loaded['indLastAvailData'].tolist())])
overall_result_for_plot_ratios = overall_result_for_plot_ratios[abs(overall_result_for_plot_ratios["ratio"]) < 300]
#
# overall_result_for_plot = pd.DataFrame({'I':overall_result_loaded['I'].tolist()}, index = overall_result_loaded['t'].tolist())
#
overall_true_I = pd.read_csv('./Data/overall_true_I-' + state_name + '.csv')
#
true_I = pd.DataFrame({'I':overall_true_I['true_I'].tolist()}, index = overall_true_I['t'].tolist())
#
# forecasted_I = overall_result_for_plot.loc[(overall_result_for_plot.index > indLastAvailData)]
#
## plot works: legend inside and outside of the figure box (we will crop the figure)
#
# plt.figure(figsize=(4,4))
true_I_with_label = true_I
true_I_with_label['source'] = "True_I"
true_I_with_label['time'] = true_I_with_label.index
true_I_with_label['Imultiplied'] = true_I_with_label["I"]*10000
# forecasted_I_with_label = forecasted_I
# forecasted_I_with_label['source'] = "Forecasted_I"
# true_and_forecasted_I = true_I_with_label
# true_and_forecasted_I = true_and_forecasted_I.append(forecasted_I, sort=False)
# true_and_forecasted_I['time'] = true_and_forecasted_I.index
# curr_relplot = sns.relplot(data=true_and_forecasted_I, x="time", y = "I", hue="source", kind="line", palette=["dimgray", "r"], legend=True)
# plt.figure(figsize=(8,5))
# fig, ax = plt.subplots()
sns.violinplot(x='indLastAvailData',y='A',data=overall_result_for_plot, palette='rainbow', hue="Payoff Matrix Entry")
plt.xlabel('Date', fontsize=14)
plt.ylabel('Matrix Entries - ' + state_name.capitalize(), fontsize=14)
# sns.violinplot(x='DateLastAvailData',y='A',data=overall_result_for_plot, palette='rainbow', hue="Payoff Matrix Entry")
# sns.violinplot(x='DateLastAvailData',y='ratio',data=overall_result_for_plot_ratios, palette='rainbow', hue="Ratio")
# sns.pointplot(x='DateLastAvailData',y='ratio',data=overall_result_for_plot_ratios, palette='rainbow', hue="Ratio", ax=ax)
#
# sns.pointplot(x='DateLastAvailData',y='ratio',data=overall_result_for_plot_ratios, palette='rainbow', hue="Ratio", ax=ax)
# curr_relplot = sns.relplot(data=true_I_with_label, x="time", y = "I", hue="source", kind="line", palette=["dimgray"], legend=False)
# curr_relplot = sns.relplot(data=true_I_with_label, x="time", y = "Imultiplied", kind="line", palette=["dimgray"], legend=False)
# curr_ax1 = curr_relplot.fig.axes[0]
# sns.pointplot(x='DateLastAvailData',y='ratio',data=overall_result_for_plot_ratios, palette='rainbow', hue="Ratio")
# curr_ax1.set_ylim(0,25)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Ratios - ' + state_name.capitalize(), fontsize=14)
# sns.scatterplot(data=true_I_with_label, x="time", y = "I", hue="source", palette=["dimgray"], legend=True, ax=ax)
# figures=[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
# plt.close(figures[0])
#
hhh = 5 + 6
# llll = [50, 100, 150, 200, 250]
# llll = list(set(overall_result_for_plot["indLastAvailData"].tolist()))
# xtick_labels = [str(i) + "::" + inpData['DateTime'].iloc[indFirstValidRow + i] for i in llll]
# xtick_labels = [inpData['DateTime'].iloc[indFirstValidRow + i] for i in llll]
# plt.xticks(llll, xtick_labels)
# plt.legend(loc='upper left')
# plt.xlabel('t', fontsize=14)
# plt.ylabel('I - ' + state_name.capitalize(), fontsize=14)
# # add the dashed line
# I_val_at_indLastAvailData = true_I.iloc[true_I.index == indLastAvailData]["I"].mean()
# plt.plot([indLastAvailData,indLastAvailData],[I_val_at_indLastAvailData - 0.001, I_val_at_indLastAvailData + 0.001], '--k')
# #
# curr_ax = plt.gcf()
# fig_name = state_name + '_' + str(indFirstAvailData) + '_' + str(indLastAvailData) + '_' + str(
#     indFirstValidRow) + '_' + str(indLastValidRow)
# with open('savedFigs/' + fig_name, 'wb') as f:
#     pickle.dump(curr_ax, f)
# plt.close()
hhh = 5+6
#
5 + 6