import pandas as pd
import numpy as np
from DataPrepTexas import dataPreparation
from DataPrepTexasMandateFree import dataPreparation_mandateFree
from FittingModel import fittingModel
from fittingModel_MF import fittingModel_MF
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import pickle
import scipy.signal
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import odeint
#from SEIRsolver import SEIR
import SEIRsolver
import SEIRsolver_UF
from numpy import linalg as LA
from tqdm import tqdm
from scipy import interpolate

use_weights_in_fitting = False
#
state_name = 'texas'
exec("from DataPrep" + state_name.capitalize() + " import dataPreparation")
#
#
#
#
I2HospitalRate_range = [0.03, 0.07, 0.09]
E2Irate_range = [1/6, 1/7, 1/9] # 1/7
I2Rrate_range = np.arange(0.11, 0.19).tolist()  #   0.154
delay_from_hospital_to_I_range = [4, 9, 16]
Beta_coeff_summer_range =  [0.085, 0.1, 0.15]  # 0.1
Beta_coeff_summer_range =  [0.15]  # 0.1 Fix these guys.

winter_coeff_range = [0.5, 0.7]   #  (1-0.421)
winter_coeff_range = [0.7]   #  (1-0.421)
Beta_coeff_manual_setting_range = [0.55] # 0.55 Fix these guys.
indFirstValidRow_range = [28]
indFirstAvailData_range = [60]
indLastAvailData_range = [100]
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
indLastAvailData_range = pd.DataFrame({'indLastAvailData_range':indLastAvailData_range}, dtype='int32')
indLastValidRow_range = pd.DataFrame({'indLastValidRow_range':indLastValidRow_range}, dtype='int32')
lower___for_sample_weights_range = pd.DataFrame({'lower___for_sample_weights_range':lower___for_sample_weights_range})
upper___for_sample_weights_range = pd.DataFrame({'upper___for_sample_weights_range':upper___for_sample_weights_range})
N0_range = pd.DataFrame({'N0_range':N0_range}, dtype='int32')
S0_range = pd.DataFrame({'S0_range':S0_range}, dtype='int32')
x0_range = pd.DataFrame({'x0_range':x0_range})
n0_range = pd.DataFrame({'n0_range':n0_range})
epsil_range = pd.DataFrame({'epsil_range':epsil_range})
## ##
#
df = I2HospitalRate_range;    df['tmp'] = 1
E2Irate_range['tmp']                    = 1; df = df.merge(E2Irate_range)
I2Rrate_range['tmp']                    = 1; df = df.merge(I2Rrate_range)
delay_from_hospital_to_I_range['tmp']   = 1; df = df.merge(delay_from_hospital_to_I_range)
Beta_coeff_summer_range['tmp']          = 1; df = df.merge(Beta_coeff_summer_range)
winter_coeff_range['tmp']               = 1; df = df.merge(winter_coeff_range)
Beta_coeff_manual_setting_range['tmp']  = 1; df = df.merge(Beta_coeff_manual_setting_range)
indFirstValidRow_range['tmp']           = 1; df = df.merge(indFirstValidRow_range)
indFirstAvailData_range['tmp']          = 1; df = df.merge(indFirstAvailData_range)
indLastAvailData_range['tmp']           = 1; df = df.merge(indLastAvailData_range)
indLastValidRow_range['tmp']            = 1; df = df.merge(indLastValidRow_range)
lower___for_sample_weights_range['tmp'] = 1; df = df.merge(lower___for_sample_weights_range)
upper___for_sample_weights_range['tmp'] = 1; df = df.merge(upper___for_sample_weights_range)
N0_range['tmp']                         = 1; df = df.merge(N0_range)
S0_range['tmp']                         = 1; df = df.merge(S0_range)
x0_range['tmp']                         = 1; df = df.merge(x0_range)
n0_range['tmp']                         = 1; df = df.merge(n0_range)
epsil_range['tmp']                      = 1; df = df.merge(epsil_range)
#
df = df.drop('tmp', axis=1)
#
df.to_csv('./Data/parameter_samples_' + state_name + '.csv')
#
print(df.shape)
#
num_of_invalid_cases = 0
#
df = df.sample(frac=1, random_state=10).reset_index(drop=True) # shit

num_parameterSamples = df.shape[0]
for i in range(0,num_parameterSamples):
# for i in range(0,100):
    if i>0: # we need 1 A, so we select the first one
        break

    # different parameter samples lead to different A's? 
    print("curr_progress = {}%".format(100*i/num_parameterSamples))
    curr_parameters = df.iloc[i].tolist()
    curr_I2HospitalRate ,curr_E2Irate, curr_I2Rrate, curr_delay_from_hospital_to_I, curr_Beta_coeff_summer, \
    curr_winter_coeff, curr_Beta_coeff_manual_setting, indFirstValidRow, indFirstAvailData, indLastAvailData, \
    indLastValidRow, lower___for_sample_weights, upper___for_sample_weights, N0, S0, x0, n0, epsil = curr_parameters
    #
    curr_I2Rrate=0.19 # fix this, this ensures we get beta_coeff/i2r = 2.02somethn
    recovRate=0
    curr_delay_from_hospital_to_I = int(curr_delay_from_hospital_to_I)
    indFirstValidRow = int(indFirstValidRow)
    indFirstAvailData = int(indFirstAvailData)
    indLastAvailData = int(indLastAvailData)
    indLastValidRow = int(indLastValidRow)
    # curr_Beta_coeff_winter = curr_Beta_coeff_summer / curr_winter_coeff
    #
    dataPreparation_mandateFree(state_name, curr_I2HospitalRate, curr_E2Irate, curr_I2Rrate, curr_delay_from_hospital_to_I, N0, S0, n0,
                    curr_Beta_coeff_summer, curr_winter_coeff, epsil)
    #
    # The true_I is the I value we get from the dataset, using our parameter combinations
    curr_InpData = pd.read_csv('./Data/NEW-' + state_name + '-history-Final-Data-with-seasonality-and-sinusoid.csv')
    if i == 0:
        # curr_true_I = pd.DataFrame({'t': curr_InpData["I"].index, 'true_I': curr_InpData["I"]/N0_range["N0_range"].iloc[0]})
        curr_true_I = pd.DataFrame({'t': curr_InpData["I"].index, 'true_I': curr_InpData["I"] / N0_range["N0_range"].iloc[0], 'true_Hospitalization':curr_InpData["hospitalizedCurrently"] / N0_range["N0_range"].iloc[0]})
        # curr_true_I = curr_InpData["I"]/N0_range["N0_range"].iloc[0]
        overall_true_I = curr_true_I
    else:
        # curr_true_I = curr_InpData["I"] / N0_range["N0_range"].iloc[0]
        curr_true_I = pd.DataFrame({'t': curr_InpData["I"].index, 'true_I': curr_InpData["I"] / N0_range["N0_range"].iloc[0], 'true_Hospitalization':curr_InpData["hospitalizedCurrently"] / N0_range["N0_range"].iloc[0]})
        overall_true_I = pd.concat([overall_true_I, curr_true_I])
    #
    # FORECAST 
    # RUN Regression, learn current A_FD and A_FR and simulate forward
    curr_Forecated_I, curr_t_modified, forecatedI_cont, t_modified_cont, inpdataY, curr_payoff_matrices_A_fd_and_A_fd = fittingModel_MF(state_name, indFirstValidRow, indFirstAvailData, indLastAvailData, indLastValidRow, curr_Beta_coeff_manual_setting, lower___for_sample_weights, upper___for_sample_weights)
    ## ## ##
    # reconstruct plot here #####################
    if False:
        fig, ax = plt.subplots(dpi=60)
        xlabel('t')
        ylabel('I')
        lx = indFirstValidRow
        rx = indLastValidRow + 30 - indFirstValidRow
        plt.plot(t_modified_cont, forecatedI_cont, 'b', linewidth=2, label='Forecasted I')
        # plt.plot(t_modified[lx:rx], Forecated_I[lx:rx], 'b', linewidth=2, label='Forecasted I')
        plt.plot(list(range(indFirstValidRow, indLastValidRow)), inpdataY, 'r', linewidth=2, label='True I')
        plt.plot([indLastAvailData, indLastAvailData], [0, 0.02], '--k')
        plt.plot([indFirstAvailData, indFirstAvailData], [0, 0.02], '--k')
        plt.plot([indLastAvailData, indLastAvailData], [0, 0.02], '--k')
        plt.plot([indFirstAvailData, indFirstAvailData], [0, 0.02], '--k')
        legend()
        llll = [50, 100, 150, 200, 250]
        ax.set_xticks(llll)
        # ax.set_xticklabels([str(i) + "::" + inpData['DateTime'].iloc[indFirstValidRow + i] for i in llll])
        ax.set_xticklabels([str(i) for i in llll])
        ax.set_xlim([indFirstValidRow, indLastValidRow+30])
        title(state_name.capitalize())
        plt.grid()
        plt.show()
#   # #################### #####################
    ## ## ##
    curr_A_fd = curr_payoff_matrices_A_fd_and_A_fd[0] # fixed now, thankfully
    curr_A_fr = curr_payoff_matrices_A_fd_and_A_fd[1] # fixed now, thankfully
    curr_A_fd_00_minus_A_fd_10 = curr_A_fd[0][0] - curr_A_fd[1][0]
    curr_A_fd_01_minus_A_fd_11 = curr_A_fd[0][1] - curr_A_fd[1][1]
    curr_ratio_A_fd = curr_A_fd_00_minus_A_fd_10 / curr_A_fd_01_minus_A_fd_11
    #
    curr_A_fr_10_minus_A_fr_00 = curr_A_fr[1][0] - curr_A_fr[0][0]
    curr_A_fr_11_minus_A_fr_01 = curr_A_fr[1][1] - curr_A_fr[0][1]
    curr_ratio_A_fr = curr_A_fr_10_minus_A_fr_00 / curr_A_fr_11_minus_A_fr_01
    #
    [curr_A_fd, curr_A_fr] = curr_payoff_matrices_A_fd_and_A_fd
    #
    if (max(curr_Forecated_I) < 0.1) and (min(curr_Forecated_I) >= 0): # and (abs(curr_ratio_A_fd) < 100) and (abs(curr_ratio_A_fr) < 100):
        try:
            curr_results = pd.DataFrame({'t': curr_t_modified, 'I': curr_Forecated_I, 'A_fd_00':curr_A_fd[0][0], 'A_fd_01':curr_A_fd[0][1], 'A_fd_10':curr_A_fd[1][0], 'A_fd_11':curr_A_fd[1][1], 'A_fr_00':curr_A_fr[0][0], 'A_fr_01':curr_A_fr[0][1], 'A_fr_10':curr_A_fr[1][0], 'A_fr_11':curr_A_fr[1][1]})
            curr_results["Hospitalized_forecast"] = curr_results['I'].multiply(curr_I2HospitalRate)
            curr_results["Hospitalized_forecast"] = curr_results["Hospitalized_forecast"].shift(periods=1 * curr_delay_from_hospital_to_I)
            # curr_results = curr_results.set_index('t')
            # overall results kind of preserves the current results for the entire band 
            overall_results = pd.concat([overall_results, curr_results])
        except NameError:
            curr_results = pd.DataFrame({'t':curr_t_modified,'I':curr_Forecated_I, 'A_fd_00':curr_A_fd[0][0], 'A_fd_01':curr_A_fd[0][1], 'A_fd_10':curr_A_fd[1][0], 'A_fd_11':curr_A_fd[1][1], 'A_fr_00':curr_A_fr[0][0], 'A_fr_01':curr_A_fr[0][1], 'A_fr_10':curr_A_fr[1][0], 'A_fr_11':curr_A_fr[1][1]})
            curr_results["Hospitalized_forecast"] = curr_results['I'].multiply(curr_I2HospitalRate)
            curr_results["Hospitalized_forecast"] = curr_results["Hospitalized_forecast"].shift(periods=1 * curr_delay_from_hospital_to_I)
            overall_results = curr_results
            # overall_results = overall_results.set_index('t')
            #
    else: # invalid case
        num_of_invalid_cases += 1
        if num_of_invalid_cases == 1:
            curr_invalid_case = df.iloc[i]
            invalid_cases = curr_invalid_case
        else:
            curr_invalid_case = df.iloc[i].T
            invalid_cases = pd.concat([invalid_cases, curr_invalid_case], axis=1)
    x = 5+6
    # now, generate a forecast for the curr_A_Fd and curr_A_Fr

#
overall_results = overall_results.set_index('t')
#
## ## ## Saving the results ## ## ##


## ## ## Inverse Validation of Methodology ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

## ## A particular forecast ## ##

norm_trends=[]
for iter in tqdm(range(50)):

    unique_forecast = overall_results.iloc[range(0,294)] # select the first forecast
    # since I do not care about the previous timeframe, let's eliminate that
    # unique_forecast = overall_results.iloc[range(72,294)]
    # this forecast is for the first combination of parameters
    # add some iid noise 
    unique_forecast['I_data'] = np.random.normal(0,0.0000001,len(unique_forecast)) + unique_forecast['I'] # rectify, shape issue, need in vector/list form DONE
    unique_forecast['I_data'] = unique_forecast['I_data'] * N0
    # this amount of noise should not interfere with the general trend of the forecast
    # we can use any unique forecast
    # the particular forecast starts at time which is the start time of the forecast (30 days)

    ## ## ## Some data preprocessing ## ## ##

    # apply savgol filter, create new I and dI
    unique_forecast['dI_data'] = unique_forecast['I_data'].diff()
    # unique_forecast['I'] = scipy.signal.savgol_filter(unique_forecast['I_data'], 11, 3)
    unique_forecast['I'] = unique_forecast['I_data']

    # the problem is, for the stitched ground truth, savgol gives negatives
    unique_forecast['I'] = unique_forecast['I'].apply(np.floor)
    unique_forecast['dI'] = unique_forecast['I'].diff()

    # adding E and dE
    I2Rrate = df['I2Rrate_range'].iloc[0] # we have the first parameter combination
    E2Irate = df['E2Irate_range'].iloc[0] # we have the first parameter combination
    unique_forecast['E'] = (unique_forecast['dI'] + I2Rrate * unique_forecast['I']).divide(E2Irate)
    unique_forecast['dE'] = unique_forecast['E'].diff()
    # Adding the terms Beta_I_S, dS, and S
    unique_forecast['Beta_I_S'] = unique_forecast['dE'] + E2Irate * unique_forecast['E']
    unique_forecast['dS'] = -1 * unique_forecast['Beta_I_S']
    # Populate S
    unique_forecast['S'] = np.nan
    startIntegrating = False
    for currIndex in range(len(unique_forecast)):
        if startIntegrating:
            unique_forecast['S'].iloc[currIndex] = unique_forecast['S'].iloc[currIndex - 1] + unique_forecast['dS'].iloc[currIndex]
        #
        if not startIntegrating:
            if np.isnan(unique_forecast['dS'].iloc[currIndex]) and (not np.isnan(unique_forecast['dS'].iloc[currIndex + 1])):
                unique_forecast['S'].iloc[currIndex] = df['S0_range'].iloc[0]
                startIntegrating = True
    # Add Beta
    unique_forecast['Beta_not_smooth'] = np.nan
    unique_forecast['Beta_not_smooth'].iloc[2:] = N0 * unique_forecast['Beta_I_S'].iloc[2:] / (unique_forecast['I'].iloc[2:] * unique_forecast['S'].iloc[2:])
    unique_forecast['Beta'] = np.nan
    # unique_forecast['Beta'].iloc[2:] = scipy.signal.savgol_filter(unique_forecast['Beta_not_smooth'].iloc[2:], 17, 3)
    unique_forecast['Beta'].iloc[2:] = unique_forecast['Beta_not_smooth'].iloc[2:]
    # Add dBeta
    unique_forecast['dBeta'] = unique_forecast['Beta'].diff()
    # Adjust Beta coefficient for summer and winter. Seasonality and Mandate Corrections in Beta

    def calc_Beta_coeff(uf):
        ind = np.asarray(uf.index)
        Beta_coeff_summer = df['Beta_coeff_summer_range'].iloc[0]
        Beta_coeff_winter = df['winter_coeff_range'].iloc[0]
        # curr_Beta = Beta_coeff_summer * (1 + (Beta_coeff_winter - Beta_coeff_summer) * np.cos(
        #         math.pi * (ind - 320)/(2 * 365)))
        curr_Beta = (Beta_coeff_summer+Beta_coeff_winter)/2 + 0.5*(Beta_coeff_winter - Beta_coeff_summer)\
            * np.cos(2*math.pi*(ind-320)/365)
        curr_Beta = (Beta_coeff_summer+Beta_coeff_winter)/2 + 0.5*(Beta_coeff_winter - Beta_coeff_summer)
        # curr_Beta = (Beta_coeff_summer+Beta_coeff_winter)/2 + 0.5*(Beta_coeff_winter - Beta_coeff_summer)\
        #     * abs(np.cos(2*math.pi*(ind-28)/365))
        # np.array(unique_forecast.index).astype(float)
        # REcover this part
        # Remove seasonality for now
        # for indx in ind:
        #     if indx<59:
        #         curr_Beta[indx-28] = 0.5 * curr_Beta[indx-28]  # first lockdown
        #     elif indx>=59 and indx<104:
        #         curr_Beta[indx-28] = curr_Beta[indx-28]  # re-opening
        #     elif indx>=104 and indx<120:
        #         curr_Beta[indx-28] = 0.6 * curr_Beta[indx-28]  # partial lockdown
        #     elif indx>=120:
        #         curr_Beta[indx-28] = 0.8 * curr_Beta[indx-28]  # mask mandate
        #     else:
        #         curr_Beta[indx-28] = curr_Beta[indx-28]
        return curr_Beta

    # Beta_coeff_summer = df['Beta_coeff_summer_range'].iloc[0]
    # Beta_coeff_winter = df['Beta_coeff_winter_range'].iloc[0]
    # curr_Beta = Beta_coeff_summer * (1 + (Beta_coeff_winter - Beta_coeff_summer) * np.cos(
    #             math.pi * (unique_forecast['t'] - 320) / (2 * 365)))

    # curr_Beta = unique_forecast.apply(calc_Beta_coeff)
    curr_Beta = calc_Beta_coeff(unique_forecast)
    unique_forecast['Beta_coeff'] = curr_Beta
    # Adding dx
    unique_forecast['dx_not_smooth'] = np.nan
    unique_forecast['dx_not_smooth'].iloc[3:] = -1 * (1/unique_forecast['Beta_coeff'].iloc[3:]) * unique_forecast['dBeta'].iloc[3:]
    unique_forecast['dx'] = np.nan
    # unique_forecast['dx'].iloc[3:] = scipy.signal.savgol_filter(unique_forecast['dx_not_smooth'].iloc[3:], 27, 3)
    unique_forecast['dx'].iloc[3:] = unique_forecast['dx_not_smooth'].iloc[3:]
    # Adding x
    unique_forecast['x'] = 1 - (1 / unique_forecast['Beta_coeff']) * unique_forecast['Beta']
    # calculating dn / dt
    unique_forecast['dn'] = (-1) * epsil * unique_forecast['dI'] / N0
    # Adding n
    unique_forecast['n'] = n0 - epsil * unique_forecast['I'] / N0

    ## ## ## Learn A_FD and A_FR again  ## ## ##
    norm_A_list=[]
    norm_A_FD_list=[]
    norm_A_FR_list=[]
    indFirstAvailData_forecast=100
    for i in range(10):
        indLastAvailData_forecast = indFirstAvailData_forecast+(i+1)*10
        ValidData = unique_forecast.iloc[indFirstAvailData_forecast-28:indLastAvailData_forecast-28]
        y_new_for_A = ValidData['dx']
        ValidData['dn'] = (-1) * epsil * unique_forecast['dI'].iloc[indFirstAvailData_forecast-28:indLastAvailData_forecast-28] / N0
        ValidData['n'] = n0 - epsil * unique_forecast['I'].iloc[indFirstAvailData_forecast-28:indLastAvailData_forecast-28] / N0
        X_new_for_A = pd.DataFrame()
        X_new_for_A['A_FD_11'] = (1 - ValidData['n']) * (ValidData['x'] ** 2 - ValidData['x'] ** 3)
        X_new_for_A['A_FD_12'] = (1 - ValidData['n']) * (ValidData['x'] - 2 * ValidData['x'] ** 2 + ValidData['x'] ** 3)
        X_new_for_A['A_FD_21'] = (1 - ValidData['n']) * (-   ValidData['x'] ** 2 + ValidData['x'] ** 3)
        X_new_for_A['A_FD_22'] = (1 - ValidData['n']) * (-ValidData['x'] + 2 * ValidData['x'] ** 2 - ValidData['x'] ** 3)
        #
        X_new_for_A['A_FR_11'] = ValidData['n'] * (ValidData['x'] ** 2 - ValidData['x'] ** 3)
        X_new_for_A['A_FR_12'] = ValidData['n'] * (ValidData['x'] - 2 * ValidData['x'] ** 2 + ValidData['x'] ** 3)
        X_new_for_A['A_FR_21'] = ValidData['n'] * (-   ValidData['x'] ** 2 + ValidData['x'] ** 3)
        X_new_for_A['A_FR_22'] = ValidData['n'] * (-ValidData['x'] + 2 * ValidData['x'] ** 2 - ValidData['x'] ** 3)
        #
        if use_weights_in_fitting:
            sample_weight = np.linspace(lower___for_sample_weights, upper___for_sample_weights, len(y_new_for_A))
            regressor = LinearRegression(fit_intercept=False)
            reg = regressor.fit(X_new_for_A, y_new_for_A, sample_weight)
        else:
            regressor = LinearRegression(fit_intercept=False)
            reg = regressor.fit(X_new_for_A, y_new_for_A)
        #
            A_FD__A_FR = reg.coef_
            A_fd = [[A_FD__A_FR[0].item(), A_FD__A_FR[1].item()], [A_FD__A_FR[2].item(), A_FD__A_FR[3].item()]]
            A_fr = [[A_FD__A_FR[4].item(), A_FD__A_FR[5].item()], [A_FD__A_FR[6].item(), A_FD__A_FR[7].item()]]
            #
            x_dot_intercept = reg.intercept_

        # Calculate Frobenius norm distance between the A_fd's and A_fr's
        norm_A_FD=LA.norm(np.subtract(curr_A_fd,A_fd), ord='fro')
        norm_A_FR=LA.norm(np.subtract(curr_A_fr,A_fr), ord='fro')
        # norm_A_FD_list.append(norm_A_FD)
        # norm_A_FR_list.append(norm_A_FR)
        # norm_A_list.append(norm_A_FD+norm_A_FR)
        deltaRT0= A_fd[0][0] - A_fd[1][0]
        deltaRT0star= curr_A_fd[0][0] - curr_A_fd[1][0]
        deltaRT1=A_fr[0][0] - A_fr[1][0]
        deltaRT1star= curr_A_fr[0][0] - curr_A_fr[1][0]

        deltaSP0=A_fd[0][1] - A_fd[1][1]
        deltaSP0star= curr_A_fd[0][1] - curr_A_fd[1][1]
        deltaSP1=A_fr[0][1] - A_fr[1][1]
        deltaSP1star= curr_A_fr[0][1] - curr_A_fr[1][1]

        dist = abs(deltaRT0-deltaRT0star)+ abs(deltaRT1-deltaRT1star) + abs(deltaSP0-deltaSP0star) +abs(deltaSP1-deltaSP1star)
        norm_A_list.append(dist)
        # if (norm_A_FD+norm_A_FR>100):
        #     print(A_fd)
        #     print(A_fr)

        if (i==3 or i==5 or i==7 or i==9) and (iter==0):
            # do a forward forecast
            # build on top of the original plot you have
            stoptime = indLastValidRow - indFirstAvailData_forecast # We do forecast for all time points after the LastValidRow
            # stoptime = indLastValidRow - indLastAvailData_forecast # We do forecast for all time points after the LastValidRow
            numpoints = 1000
            tt = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
            abserr = 1.0e-8
            relerr = 1.0e-6
            # # # fix the A's. I couldn't care less about what A's you learn everytime.
            # A_fd=np.array([[-0.13825701128968554, -0.10209015455509081], [0.13825701128968726, 0.10209015455509088]])
            # A_fr=np.array([[[0.1844214096581435, 0.1335051812829616], [-0.1844214096581436, -0.13350518128296152]]])
            # DO a forward forecast
            # parameters = [A_fr, A_fd, I2Rrate, E2Irate, N0, epsil, recovRate, x_dot_intercept, unique_forecast, indLastAvailData_forecast, curr_Beta_coeff_manual_setting] # parameters for forecast
            # R0 = 1 - unique_forecast['S'].iloc[indLastAvailData_forecast-28] / N0 - unique_forecast['E'].iloc[indLastAvailData_forecast-28] / N0 - unique_forecast['I'].iloc[indLastAvailData_forecast-28] / N0
            # y0 = [unique_forecast['S'].iloc[indLastAvailData_forecast-28] / N0,
            #     unique_forecast['E'].iloc[indLastAvailData_forecast-28] / N0,
            #     unique_forecast['I'].iloc[indLastAvailData_forecast-28] / N0,
            #     R0,
            #     unique_forecast['x'].iloc[indLastAvailData_forecast-28],
            #     unique_forecast['n'].iloc[indLastAvailData_forecast-28]]
            # wsol = odeint(SEIRsolver_UF.SEIR_uf, y0, tt, args=(parameters,), atol=abserr, rtol=relerr) #ODE forward simulation
            # #
            # Forecast_I_validate = list(unique_forecast['I'].iloc[(indFirstValidRow-28):(indLastAvailData_forecast-28)] / N0) #fetch forecast
            # Forecast_I_validate.extend(wsol[:, 2]) #
            # DO a foreward forecastV2
            parameters = [A_fr, A_fd, I2Rrate, E2Irate, N0, epsil, recovRate, x_dot_intercept, unique_forecast, indFirstAvailData_forecast, curr_Beta_coeff_manual_setting] # parameters for forecast
            R0 = 1 - unique_forecast['S'].iloc[indFirstAvailData_forecast-28] / N0 - unique_forecast['E'].iloc[indFirstAvailData_forecast-28] / N0 - unique_forecast['I'].iloc[indFirstAvailData_forecast-28] / N0
            y0 = [unique_forecast['S'].iloc[indFirstAvailData_forecast-28] / N0,
                unique_forecast['E'].iloc[indFirstAvailData_forecast-28] / N0,
                unique_forecast['I'].iloc[indFirstAvailData_forecast-28] / N0,
                R0,
                unique_forecast['x'].iloc[indFirstAvailData_forecast-28],
                unique_forecast['n'].iloc[indFirstAvailData_forecast-28]]
            wsol = odeint(SEIRsolver_UF.SEIR_uf, y0, tt, args=(parameters,), atol=abserr, rtol=relerr) #ODE forward simulation
            #
            Forecast_I_validate = list(unique_forecast['I'].iloc[(indFirstValidRow-28):(indFirstAvailData_forecast-28)] / N0) #fetch forecast
            Forecast_I_validate.extend(wsol[:, 2]) #
            # # Plotting
            # t_modified = list(range(indFirstValidRow, indLastAvailData_forecast))
            # t_modified.extend([i + indLastAvailData_forecast for i in tt])
            # Plotting V2
            t_modified = list(range(indFirstValidRow, indFirstAvailData_forecast))
            t_modified.extend([i + indFirstAvailData_forecast for i in tt])
            # Integrate
            #
            Forecast_I_interpolator_fn = interpolate.interp1d(t_modified, Forecast_I_validate)
            t_modified_only_for_int_val = list(range(int(min(t_modified)), int(max(t_modified)) + 1, 1))
            Forecast_I_only_for_int_val = Forecast_I_interpolator_fn(t_modified_only_for_int_val)
            # plot
            
                
            fig, ax = plt.subplots(dpi=60)
            xlabel('t')
            ylabel('I')
            lx = indFirstValidRow
            rx = indLastValidRow + 30 - indFirstValidRow
            plt.plot(t_modified, Forecast_I_validate, 'b', linewidth=2, label='Forecast I '+str(i)+'0')
            plt.plot(t_modified_cont, forecatedI_cont, 'r', linewidth=2, label='Ground truth I')
            # plt.plot(t_modified[lx:rx], Forecated_I[lx:rx], 'b', linewidth=2, label='Forecasted I')
            # plt.plot(list(range(indFirstValidRow, indLastValidRow)), inpdataY, 'r', linewidth=2, label='True I')
            # plt.plot([indLastAvailData, indLastAvailData], [0, 0.02], '--k')
            maxx=max(max(Forecast_I_validate),max(forecatedI_cont))
            plt.plot([indFirstAvailData_forecast, indFirstAvailData_forecast], [0, maxx])
            plt.plot([indLastAvailData_forecast, indLastAvailData_forecast], [0, maxx])
            # plt.plot([indLastAvailData, indLastAvailData], [0, 0.02])
            # plt.plot([indFirstAvailData, indFirstAvailData], [0, 0.02])
            legend()
            llll = [50, 100, 150, 200, 250]
            ax.set_xticks(llll)
            # ax.set_yticks([0.0025,0.0050,0.0075,0.01,0.0125,0.0150,0.0175,0.02])
            # ax.set_xticklabels([str(i) + "::" + inpData['DateTime'].iloc[indFirstValidRow + i] for i in llll])
            ax.set_xticklabels([str(i) for i in llll])
            ax.set_xlim([indFirstValidRow, indLastValidRow+30])
            title(state_name.capitalize())
            plt.grid()
            plt.show()


    norm_trends.append(norm_A_list)

# Plot the norm trends
from matplotlib.ticker import FormatStrFormatter
fig, dia = plt.subplots(dpi=60)
n=np.arange(10,110,10)
k=0
for trend in norm_trends:
    if k==0:
        dia.plot(n,trend,'-',\
            marker='o',color='black')
        k+=1
    else:
        dia.plot(n,trend,'-',marker='o',color='black')
# dia.label='||A_FD* - A_FD||+||A_FR* - A_FR||'
dia.set_xlabel("Learning data points",style='italic',fontsize=12)
dia.set_ylabel("Average Matrix distance", style='italic',fontsize=12)
dia.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
legend = dia.legend(loc='upper right', shadow=True, fontsize=12)
plt.xticks(n)
plt.show()

# Plot the evolution of the norms
# A_FD_plot=plot(norm_A_FD_list)
# A_FR_plot=plot(norm_A_FR_list)
# A_FD_plot.savefig('savedFigs/'+'A_FD_plot'+'.png') 
# A_FR_plot.savefig('savedFigs/'+'A_FR_plot'+'.png') 

# fig, dia = plt.subplots(dpi=60)
# # n=np.array([4,8,12,16,20,24,28,32,36,40])
# n=np.arange(10,210,10)
# dia.plot(n,norm_A_FD_list,'-',label='||A_FD* - A_FD||',marker="o",color='black')
# dia.plot(n,norm_A_FR_list,'--',label='||A_FR* - A_FR||',marker="x",color='black')

# dia.set_xlabel("Learning data points",style='italic',fontsize=8)
# dia.set_ylabel("Frobenius distance", style='italic',fontsize=8)
# dia.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# legend = dia.legend(loc='upper left', shadow=True, fontsize=8)
# plt.xticks(n)
# # dia.set_title("X/Xs")
# plt.show()

# # Calculate Frobenius norm distance between the A_fd's and A_fr's
# norm_A_FD=LA.norm(np.subtract(curr_A_fd,A_fd), ord='fro')
# norm_A_FR=LA.norm(np.subtract(curr_A_fr,A_fr), ord='fro')

# ## ## ## Create New Forecast ## ## ##
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# stoptime = indLastValidRow - indLastAvailData
# numpoints = 1000
# t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
# abserr = 1.0e-8
# relerr = 1.0e-6
# #
# recovRate = 0
# parameters = [A_fr, A_fd, I2Rrate, E2Irate, N0, epsil, recovRate, x_dot_intercept, unique_forecast, indLastAvailData, df['Beta_coeff_manual_setting_range'].iloc[0]]
# R0 = 1 - unique_forecast['S'].iloc[indLastAvailData] / N0 - unique_forecast['E'].iloc[indLastAvailData] / N0 - unique_forecast['I'].iloc[indLastAvailData] / N0
# y0 = [unique_forecast['S'].iloc[indLastAvailData] / N0,
#         unique_forecast['E'].iloc[indLastAvailData] / N0,
#         unique_forecast['I'].iloc[indLastAvailData] / N0,
#         R0,
#         unique_forecast['x'].iloc[indLastAvailData],
#         unique_forecast['n'].iloc[indLastAvailData]]
# wsol = odeint(SEIRsolver.SEIR, y0, t, args=(parameters,), atol=abserr, rtol=relerr)
# #
# Forecated_I = list(unique_forecast['I'].iloc[indFirstValidRow:indLastAvailData] / N0)
# Forecated_I.extend(wsol[:, 2])
# #
# # plotting
# t_modified = list(range(indFirstValidRow, indLastAvailData))
# t_modified.extend([i + indLastAvailData for i in t])
# #
# from scipy import interpolate
# Forecated_I_interpolator_function = interpolate.interp1d(t_modified, Forecated_I)
# t_modified_only_for_integer_value = list(range(int(min(t_modified)), int(max(t_modified)) + 1, 1))
# Forecated_I_only_for_integer_value = Forecated_I_interpolator_function(t_modified_only_for_integer_value)
# #

# # payoff_matrices_A_fd_and_A_fd = [A_fd, A_fr]
# # return Forecated_I, t_modified
# # return Forecated_I_only_for_integer_value, t_modified_only_for_integer_value, payoff_matrices_A_fd_and_A_fd


