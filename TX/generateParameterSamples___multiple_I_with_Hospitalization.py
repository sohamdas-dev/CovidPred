import pandas as pd
import numpy as np
from DataPrepTexas import dataPreparation
from FittingModel import fittingModel
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
winter_coeff_range = [0.5, 0.7]   #  (1-0.421)
Beta_coeff_manual_setting_range = [0.55] # 0.55
indFirstValidRow_range = [28]
indFirstAvailData_range = [60]
indLastAvailData_range = [250]
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
df = df.sample(frac=1).reset_index(drop=True)
num_parameterSamples = df.shape[0]
for i in range(0,num_parameterSamples):
# for i in range(0,100):
    print("curr_progress = {}%".format(100*i/num_parameterSamples))
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
    curr_Forecated_I, curr_t_modified, curr_payoff_matrices_A_fd_and_A_fd = fittingModel(state_name, indFirstValidRow, indFirstAvailData, indLastAvailData, indLastValidRow, curr_Beta_coeff_manual_setting, lower___for_sample_weights, upper___for_sample_weights)
    #
    curr_A_fd = curr_payoff_matrices_A_fd_and_A_fd[0]
    curr_A_fr = curr_payoff_matrices_A_fd_and_A_fd[1]
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
#
overall_results = overall_results.set_index('t')
#
# Saving the results
# overall_results.to_csv('./Data/overall_results-' + state_name + '.csv', index=False)
# This is where we write the results out 
overall_true_I.to_csv('./Data/overall_true_I-' + state_name + '.csv')
#
overall_results.to_csv('./Data/overall_results-' + state_name + '.csv')
#
try:
    invalid_cases = invalid_cases.T
    invalid_cases.to_csv('./Data/invalid_cases-' + state_name + '.csv')
except NameError:
    invalid_cases = pd.DataFrame([])
    invalid_cases.to_csv('./Data/invalid_cases-' + state_name + '.csv')
#
hhh = 5+6

## ## ## Inverse Validation of Methodology ## ## ##
# You have overall results for the forecast
# The forecast is for 294 days = 42 weeks 
# You have I and H_forecast for the 294 days
# Now take the forecast for the real I data.
# Calculate dI from the real I data.
# Where do we get the x value/? 
# Let us take this data for given. The levels of i for the next 294 days. 
# Can we infer the value of A_FD and A_FR by running the same process?
# How much data do we need to train on to learn the parameters
# We can apply this theme, and create iterative improvement of the parameters. 
# Eventually the parameters will cease to change
# Then we have reached the optimal value of the parameters

## ## A particular forecast ## ##
unique_forecast = overall_results.iloc[range(0,294)] # select the first forecast
# this forecast is for the first combination of parameters
# add some iid noise 
# unique_forecast['I_data'] = np.random.normal(0,0.00001,[len(unique_forecast), 1]) + unique_forecast['I'] # rectify, shape issue, need in vector/list form
unique_forecast['I_data'] = unique_forecast['I']
# this amount of noise should not interfere with the general trend of the forecast
# we can use any unique forecast
# the particular forecast starts at time which is the start time of the forecast (30 days)

## ## ## Some data preprocessing ## ## ##

# apply savgol filter, create new I and dI
unique_forecast['dI_data'] = unique_forecast['I_data'].diff()
unique_forecast['I'] = scipy.signal.savgol_filter(unique_forecast['I_data'], 51, 3)
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
for i in range(len(unique_forecast)):
    if i==0:
        unique_forecast['S'].iloc[i] = df['S0_range'].iloc[0] # we have the first parameter combination
    else:
        unique_forecast['S'].iloc[i] = unique_forecast['S'].iloc[i-1] + unique_forecast['dS'].iloc[i-1]
# Add Beta
unique_forecast['Beta_not_smooth'] = np.nan
unique_forecast['Beta_not_smooth'] = N0 * unique_forecast['Beta_I_S'] / (unique_forecast['I'] * unique_forecast['S'])
unique_forecast['Beta'] = np.nan
unique_forecast['Beta'] = scipy.signal.savgol_filter(unique_forecast['Beta_not_smooth'].iloc[2:], 17, 3)
# Add dBeta
unique_forecast['dBeta'] = unique_forecast['Beta'].diff()
# Adjust Beta coefficient for summer and winter
def calc_Beta_coeff(unique_forecast):
    Beta_coeff_summer = df['Beta_coeff_summer_range'].iloc[0]
    Beta_coeff_winter = df['Beta_coeff_winter_range'].iloc[0]
    curr_Beta = Beta_coeff_summer * (1 + (Beta_coeff_winter - Beta_coeff_summer) * np.cos(
            math.pi * (unique_forecast['t'] - 320) / (2 * 365)))
    if unique_forecast['t']<59:
        curr_Beta = 0.5 * curr_Beta  # first lockdown
    elif unique_forecast['t']>=59 and unique_forecast['t']<104:
        curr_Beta = curr_Beta  # re-opening
    elif unique_forecast['t']>=104 and unique_forecast['t']<120:
        curr_Beta = 0.6 * curr_Beta  # partial lockdown
    elif unique_forecast['t']>=120:
        curr_Beta = 0.8 * curr_Beta  # mask mandate
    else:
        curr_Beta = curr_Beta
    return curr_Beta

unique_forecast['Beta_coeff'] = unique_forecast.apply(calc_Beta_coeff, axis=1)
# Adding dx
unique_forecast['dx_not_smooth'] = np.nan
unique_forecast['dx_not_smooth'] = -1 * (1 / unique_forecast['Beta_coeff']) * unique_forecast['dBeta']
unique_forecast['dx'] = np.nan
unique_forecast['dx'] = scipy.signal.savgol_filter(unique_forecast['dx_not_smooth'], 27, 3)
# Adding x
unique_forecast['x'] = 1 - (1 / unique_forecast['Beta_coeff']) * unique_forecast['Beta']
# calculating dn / dt
unique_forecast['dn'] = (-1) * epsil * unique_forecast['dI'] / N0
# Adding n
unique_forecast['n'] = n0 - epsil * unique_forecast['I'] / N0

## ## ## Learn A_FD and A_FR again  ## ## ##


ValidData = unique_forecast.iloc[indFirstAvailData:indLastAvailData]
y_new_for_A = ValidData['dx']
ValidData['dn'] = (-1) * epsil * unique_forecast['dI'].iloc[indFirstAvailData:indLastAvailData] / N0
ValidData['n'] = n0 - epsil * unique_forecast['I'].iloc[indFirstAvailData:indLastAvailData] / N0
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


## ## ## Create New Forecast ## ## ##
stoptime = indLastValidRow - indLastAvailData
numpoints = 1000
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
abserr = 1.0e-8
relerr = 1.0e-6
#
recovRate = 0
parameters = [A_fr, A_fd, I2Rrate, E2Irate, N0, epsil, recovRate, x_dot_intercept, unique_forecast, indLastAvailData, df['Beta_coeff_manual_setting_range'].iloc[0]]
R0 = 1 - unique_forecast['S'].iloc[indLastAvailData] / N0 - unique_forecast['E'].iloc[indLastAvailData] / N0 - unique_forecast['I'].iloc[indLastAvailData] / N0
y0 = [unique_forecast['S'].iloc[indLastAvailData] / N0,
        unique_forecast['E'].iloc[indLastAvailData] / N0,
        unique_forecast['I'].iloc[indLastAvailData] / N0,
        R0,
        unique_forecast['x'].iloc[indLastAvailData],
        unique_forecast['n'].iloc[indLastAvailData]]
wsol = odeint(SEIRsolver.SEIR, y0, t, args=(parameters,), atol=abserr, rtol=relerr)
#
Forecated_I = list(unique_forecast['I'].iloc[indFirstValidRow:indLastAvailData] / N0)
Forecated_I.extend(wsol[:, 2])
#
# plotting
t_modified = list(range(indFirstValidRow, indLastAvailData))
t_modified.extend([i + indLastAvailData for i in t])
#
from scipy import interpolate
Forecated_I_interpolator_function = interpolate.interp1d(t_modified, Forecated_I)
t_modified_only_for_integer_value = list(range(int(min(t_modified)), int(max(t_modified)) + 1, 1))
Forecated_I_only_for_integer_value = Forecated_I_interpolator_function(t_modified_only_for_integer_value)
#

# payoff_matrices_A_fd_and_A_fd = [A_fd, A_fr]
# return Forecated_I, t_modified
# return Forecated_I_only_for_integer_value, t_modified_only_for_integer_value, payoff_matrices_A_fd_and_A_fd
