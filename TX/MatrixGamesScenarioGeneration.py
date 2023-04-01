import pandas as pd
import numpy as np
from DataPrepTexas import dataPreparation
from FittingModel import fittingModel
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import pickle
from Calc_payoff_matrices import payoffMatrices
#
state_name = 'texas'
exec("from DataPrep" + state_name.capitalize() + " import dataPreparation")
#
#
I2HospitalRate_range = [0.03, 0.07, 0.09]
E2Irate_range = [1/6, 1/7, 1/9] # 1/7
I2Rrate_range = np.arange(0.11, 0.19).tolist()  #   0.154
delay_from_hospital_to_I_range = [4, 9, 16]
Beta_coeff_summer_range =  [0.085, 0.1, 0.15]  # 0.1
winter_coeff_range = [0.5, 0.7]   #  (1-0.421)
Beta_coeff_manual_setting_range = [0.55] # 0.55
indFirstValidRow_range = [28]
indFirstAvailData_range = [60]
# indLastAvailData_range = [160]
indLastValidRow_range = [321]
lower___for_sample_weights_range = [1]
upper___for_sample_weights_range = [100]
N0_range = [29000000]
S0_range = N0_range
x0_range = [0.7]
n0_range = [0.6, 0.8, 0.9]
epsil_range = [30, 40]
#
## ## ##
I2HospitalRate_range = pd.DataFrame({'I2HospitalRate_range':I2HospitalRate_range})
E2Irate_range = pd.DataFrame({'E2Irate_range':E2Irate_range})
I2Rrate_range = pd.DataFrame({'I2Rrate_range':I2Rrate_range})
delay_from_hospital_to_I_range = pd.DataFrame({'delay_from_hospital_to_I_range':delay_from_hospital_to_I_range}, dtype='int32')
Beta_coeff_summer_range = pd.DataFrame({'Beta_coeff_summer_range':Beta_coeff_summer_range})
winter_coeff_range = pd.DataFrame({'winter_coeff_range':winter_coeff_range})
Beta_coeff_manual_setting_range = pd.DataFrame({'Beta_coeff_manual_setting_range':Beta_coeff_manual_setting_range})
indFirstValidRow_range = pd.DataFrame({'indFirstValidRow_range':indFirstValidRow_range}, dtype='int32')
indFirstAvailData_range = pd.DataFrame({'indFirstAvailData_range':indFirstAvailData_range}, dtype='int32')
# indLastAvailData_range = pd.DataFrame({'indLastAvailData_range':indLastAvailData_range}, dtype='int32')
indLastValidRow_range = pd.DataFrame({'indLastValidRow_range':indLastValidRow_range}, dtype='int32')
lower___for_sample_weights_range = pd.DataFrame({'lower___for_sample_weights_range':lower___for_sample_weights_range})
upper___for_sample_weights_range = pd.DataFrame({'upper___for_sample_weights_range':upper___for_sample_weights_range})
N0_range = pd.DataFrame({'N0_range':N0_range}, dtype='int32')
S0_range = pd.DataFrame({'S0_range':S0_range}, dtype='int32')
x0_range = pd.DataFrame({'x0_range':x0_range})
n0_range = pd.DataFrame({'n0_range':n0_range})
epsil_range = pd.DataFrame({'epsil_range':epsil_range})
#
#
list_of_indLastAvailData = [90, 140, 250]
#
#
for j in range(0,len(list_of_indLastAvailData)):
    #
    curr_indLastAvailData = list_of_indLastAvailData[j]
    #
    indLastAvailData_range = [curr_indLastAvailData]
    indLastAvailData_range = pd.DataFrame({'indLastAvailData_range': indLastAvailData_range}, dtype='int32')
    #
    ## ##
    #
    df = I2HospitalRate_range
    df = df.merge(E2Irate_range, how='cross')
    df = df.merge(I2Rrate_range, how='cross')
    df = df.merge(delay_from_hospital_to_I_range, how='cross')
    df = df.merge(Beta_coeff_summer_range, how='cross')
    df = df.merge(winter_coeff_range, how='cross')
    df = df.merge(Beta_coeff_manual_setting_range, how='cross')
    df = df.merge(indFirstValidRow_range, how='cross')
    df = df.merge(indFirstAvailData_range, how='cross')
    df = df.merge(indLastAvailData_range, how='cross')
    df = df.merge(indLastValidRow_range, how='cross')
    df = df.merge(lower___for_sample_weights_range, how='cross')
    df = df.merge(upper___for_sample_weights_range, how='cross')
    df = df.merge(N0_range, how='cross')
    df = df.merge(S0_range, how='cross')
    df = df.merge(x0_range, how='cross')
    df = df.merge(n0_range, how='cross')
    df = df.merge(epsil_range, how='cross')
    #
    df.to_csv('./Data/parameter_samples_' + state_name + '.csv')
    #
    print(df.shape)
    #
    num_of_invalid_cases = 0
    #
    df = df.sample(frac=1).reset_index(drop=True)
    num_parameterSamples = df.shape[0]
    for i in range(0,num_parameterSamples):
    # for i in range(0,100):
        print("curr_progress = {}%---{}%".format(100*j/len(list_of_indLastAvailData), 100*i/num_parameterSamples))
        curr_parameters = df.iloc[i].tolist()
        curr_I2HospitalRate ,curr_E2Irate, curr_I2Rrate, curr_delay_from_hospital_to_I, curr_Beta_coeff_summer, \
        curr_winter_coeff, curr_Beta_coeff_manual_setting, indFirstValidRow, indFirstAvailData, indLastAvailData, \
        indLastValidRow, lower___for_sample_weights, upper___for_sample_weights, N0, S0, x0, n0, epsil = curr_parameters
        #
        curr_delay_from_hospital_to_I = int(curr_delay_from_hospital_to_I)
        indFirstValidRow = int(indFirstValidRow)
        indFirstAvailData = int(indFirstAvailData)
        indLastAvailData = int(indLastAvailData)
        indLastValidRow = int(indLastValidRow)
        curr_Beta_coeff_winter = curr_Beta_coeff_summer / curr_winter_coeff
        #
        dataPreparation(state_name, curr_I2HospitalRate, curr_E2Irate, curr_I2Rrate, curr_delay_from_hospital_to_I, N0, S0, n0,
                        curr_Beta_coeff_summer, curr_Beta_coeff_winter, epsil)
        #
        #
        curr_InpData = pd.read_csv('./Data/NEW-' + state_name + '-history-Final-Data-with-seasonality-and-sinusoid.csv')
        if i == 0:
            curr_true_I = pd.DataFrame({'t': curr_InpData["I"].index, 'true_I': curr_InpData["I"]/N0_range["N0_range"].iloc[0]})
            # curr_true_I = curr_InpData["I"]/N0_range["N0_range"].iloc[0]
            overall_true_I = curr_true_I
        else:
            # curr_true_I = curr_InpData["I"] / N0_range["N0_range"].iloc[0]
            curr_true_I = pd.DataFrame({'t': curr_InpData["I"].index, 'true_I': curr_InpData["I"] / N0_range["N0_range"].iloc[0]})
            overall_true_I = pd.concat([overall_true_I, curr_true_I])
        #
        curr_A_fd, curr_A_fr, curr_x_dot_intercept = payoffMatrices(state_name, indFirstValidRow, indFirstAvailData, indLastAvailData, indLastValidRow, curr_Beta_coeff_manual_setting, lower___for_sample_weights, upper___for_sample_weights)
        #
        # curr_Forecated_I, curr_t_modified = fittingModel(state_name, indFirstValidRow, indFirstAvailData, indLastAvailData, indLastValidRow, curr_Beta_coeff_manual_setting, lower___for_sample_weights, upper___for_sample_weights)
        #
        [A_fd_00, A_fd_01] = curr_A_fd[0]
        [A_fd_10, A_fd_11] = curr_A_fd[1]
        [A_fr_00, A_fr_01] = curr_A_fr[0]
        [A_fr_10, A_fr_11] = curr_A_fr[1]
        #
        try:
            curr_results = pd.DataFrame({'A_fd_00': A_fd_00, 'A_fd_01': A_fd_01,
                                         'A_fd_10': A_fd_10, 'A_fd_11': A_fd_11,
                                         'A_fr_00': A_fr_00, 'A_fr_01': A_fr_01,
                                         'A_fr_10': A_fr_10, 'A_fr_11': A_fr_11,
                                         'indLastAvailData': curr_indLastAvailData}, index=[8])
            # curr_results = curr_results.set_index('indLastAvailData')
            #
            overall_results = pd.concat([overall_results, curr_results])
        except NameError:
            curr_results = pd.DataFrame({'A_fd_00': A_fd_00, 'A_fd_01': A_fd_01,
                                         'A_fd_10': A_fd_10, 'A_fd_11': A_fd_11,
                                         'A_fr_00': A_fr_00, 'A_fr_01': A_fr_01,
                                         'A_fr_10': A_fr_10, 'A_fr_11': A_fr_11,
                                         'indLastAvailData': curr_indLastAvailData}, index=[8])
            # curr_results = curr_results.set_index('indLastAvailData')
            overall_results = curr_results
        #
        hhh = 5+6
        #
        # if (max(curr_Forecated_I) < 0.3) and (min(curr_Forecated_I) >= 0):
        #     try:
        #         curr_results = pd.DataFrame({'t': curr_t_modified, 'I': curr_Forecated_I})
        #         # curr_results = curr_results.set_index('t')
        #         overall_results = pd.concat([overall_results, curr_results])
        #     except NameError:
        #         curr_results = pd.DataFrame({'t':curr_t_modified,'I':curr_Forecated_I})
        #         overall_results = curr_results
        #         # overall_results = overall_results.set_index('t')
        #         #
        # else: # invalid case
        #     num_of_invalid_cases += 1
        #     if num_of_invalid_cases == 1:
        #         curr_invalid_case = df.iloc[i]
        #         invalid_cases = curr_invalid_case
        #     else:
        #         curr_invalid_case = df.iloc[i].T
        #         invalid_cases = pd.concat([invalid_cases, curr_invalid_case], axis=1)
        # x = 5+6
#
overall_results = overall_results.set_index('indLastAvailData')
#
# Saving the results
#
# overall_true_I.to_csv('./Data/overall_true_I-' + state_name + '.csv')
#
overall_results.to_csv('./Data/Afd_Afr_matrices-' + state_name + '.csv')
#
# try:
#     invalid_cases = invalid_cases.T
#     invalid_cases.to_csv('./Data/invalid_cases-' + state_name + '.csv')
# except NameError:
#     invalid_cases = pd.DataFrame([])
#     invalid_cases.to_csv('./Data/invalid_cases-' + state_name + '.csv')
#
hhh = 5+6