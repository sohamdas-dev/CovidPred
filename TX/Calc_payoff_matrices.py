
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
from SEIRsolver import SEIR



def payoffMatrices(state_name, indFirstValidRow, indFirstAvailData, indLastAvailData, indLastValidRow, Beta_coeff_manual_setting, lower___for_sample_weights, upper___for_sample_weights):
    #
    inpData = pd.read_csv('./Data/NEW-' + state_name + '-history-Final-Data-with-seasonality-and-sinusoid.csv')
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
    X_new_for_A['A__21'] = ValidData['n'] * (-   ValidData['x'] ** 2 + ValidData['x'] ** 3)
    X_new_for_A['A_FR_22'] = ValidData['n'] * (-ValidData['x'] + 2 * ValidData['x'] ** 2 - ValidData['x'] ** 3)
    #
    sample_weight = np.linspace(lower___for_sample_weights, upper___for_sample_weights, len(y_new_for_A))
    regressor = LinearRegression(fit_intercept=False)
    reg = regressor.fit(X_new_for_A, y_new_for_A, sample_weight)
    #
    A_FD__A_FR = reg.coef_
    A_fd = [[A_FD__A_FR[0].item(), A_FD__A_FR[1].item()], [A_FD__A_FR[2].item(), A_FD__A_FR[3].item()]]
    A_fr = [[A_FD__A_FR[4].item(), A_FD__A_FR[5].item()], [A_FD__A_FR[6].item(), A_FD__A_FR[7].item()]]
    #
    x_dot_intercept = reg.intercept_
    #
    #
    return A_fd, A_fr, x_dot_intercept







