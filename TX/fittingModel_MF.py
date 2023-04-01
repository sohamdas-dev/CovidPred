
# this file is created based on the file: FittingTheModelCalifornia-with-epsilon-and-seasonality-sinusoid-with-Beta_adjustment.ipynb

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import scipy.signal
from scipy.optimize import curve_fit
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import odeint
#from SEIRsolver import SEIR
import SEIRsolver
import statsmodels
import statsmodels.api as sm

use_weights_in_fitting = False

def fittingModel_MF(state_name, indFirstValidRow, indFirstAvailData, indLastAvailData, indLastValidRow, Beta_coeff_manual_setting, lower___for_sample_weights, upper___for_sample_weights):
    #
    inpData = pd.read_csv('./Data/NEW-' + state_name + '-history-Final-Data-with-seasonality-and-sinusoidV2.csv')
    #
    parameters_input = []
    with open('./Data/parameters-wiith-seasonality.txt', 'r') as filehandle:
        for line in filehandle:
            parameters_input.append(float(line))
    #
    E2Irate, I2Rrate, Beta_coeff_summer, Beta_coeff_winter, epsil, N0, n0 = parameters_input
    recovRate = 0
    #
    dict_Dates_to_index = dict()
    dict_index_to_Dates = dict()
    for i in inpData.index:
        dict_Dates_to_index[inpData['DateTime'].iloc[i]] = i
        dict_index_to_Dates[str(i)] = inpData['DateTime'].iloc[i]
    # 
    #
    # fitting the model
    ValidData = inpData.iloc[indFirstAvailData:indLastAvailData]
    y_new_for_A = ValidData['dx']
    ValidData['dn'] = (-1) * epsil * inpData['dI'].iloc[indFirstAvailData:indLastAvailData] / N0
    ValidData['n'] = n0 - epsil * inpData['I'].iloc[indFirstAvailData:indLastAvailData] / N0
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
    
    ## ## ## Start the forecast ## ## ##
    stoptime = indLastValidRow - indLastAvailData # We do forecast for all time points after the LastValidRow
    numpoints = 1000
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    abserr = 1.0e-8
    relerr = 1.0e-6
    # # fix the A's. I couldn't care less about what A's you learn everytime.
    A_fd=[[3, 1], [3.2, 0.25]]
    A_fr=[[0.5, 1], [1, 1.25]]

    parameters = [A_fr, A_fd, I2Rrate, E2Irate, N0, epsil, recovRate, x_dot_intercept, inpData, indLastAvailData, Beta_coeff_manual_setting] # parameters for forecast
    R0 = 1 - inpData['S'].iloc[indLastAvailData] / N0 - inpData['E'].iloc[indLastAvailData] / N0 - inpData['I'].iloc[indLastAvailData] / N0
    y0 = [inpData['S'].iloc[indLastAvailData] / N0,
          inpData['E'].iloc[indLastAvailData] / N0,
          inpData['I'].iloc[indLastAvailData] / N0,
          R0,
          inpData['x'].iloc[indLastAvailData],
          inpData['n'].iloc[indLastAvailData]]
    wsol = odeint(SEIRsolver.SEIR, y0, t, args=(parameters,), atol=abserr, rtol=relerr) #ODE forward simulation
    #
    Forecated_I = list(inpData['I'].iloc[indFirstValidRow:indLastAvailData] / N0) #fetch forecast
    Forecated_I.extend(wsol[:, 2]) #
    #
    
    
    # plotting
    t_modified = list(range(indFirstValidRow, indLastAvailData))
    t_modified.extend([i + indLastAvailData for i in t])
    #
    from scipy import interpolate
    Forecated_I_interpolator_function = interpolate.interp1d(t_modified, Forecated_I)
    t_modified_only_for_integer_value = list(range(int(min(t_modified)), int(max(t_modified)) + 1, 1))
    Forecated_I_only_for_integer_value = Forecated_I_interpolator_function(t_modified_only_for_integer_value)
    
    # so this is where the forecast is printed
    if False:
        date_modified = inpData['DateTime']
        # figure(1, figsize=(8, 6.5))
        # fig, ax = plt.subplots(figsize=(12, 9.5))
        fig, ax = plt.subplots(dpi=60)
        xlabel('t')
        ylabel('I')
        lx = indFirstValidRow
        rx = indLastValidRow + 30 - indFirstValidRow
        plt.plot(t_modified, Forecated_I, 'b', linewidth=2, label='Forecasted I')
        # plt.plot(t_modified[lx:rx], Forecated_I[lx:rx], 'b', linewidth=2, label='Forecasted I')
        plt.plot(list(range(indFirstValidRow, indLastValidRow)), inpData['I'].iloc[indFirstValidRow:indLastValidRow] / N0, 'r', linewidth=2, label='True I')
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
    #
    payoff_matrices_A_fd_and_A_fd = [A_fd, A_fr]
    
    # return Forecated_I, t_modified
    return Forecated_I_only_for_integer_value, t_modified_only_for_integer_value, \
         Forecated_I, t_modified, inpData['I'].iloc[indFirstValidRow:indLastValidRow] / N0, \
             payoff_matrices_A_fd_and_A_fd







