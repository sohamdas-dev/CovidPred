# DataprepTexas forecast 

# this file is created based on the file: DataPreparationColorado-withSeasonality-sinusoid-BetaCoeff.ipynb

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import scipy.signal
import math



def dataPreparation_forecast(state_name, I2HospitalRate, E2Irate, I2Rrate, delay_from_hospital_to_I, N0, S0, n0,
                    Beta_coeff_summer_input, Beta_coeff_winter_input, epsil):
    #
    global Beta_coeff_summer
    global Beta_coeff_winter 
    Beta_coeff_summer = Beta_coeff_summer_input
    Beta_coeff_winter = Beta_coeff_winter_input
    #
    with open('./Data/parameters-wiith-seasonality.txt', 'w') as filehandle:
        for listitem in [E2Irate, I2Rrate, Beta_coeff_summer, Beta_coeff_winter, epsil, N0, n0]:
            filehandle.write('%s\n' % listitem)
    # Read the forecast and 
    inpData = pd.read_csv('./Data/' + state_name + '-history.csv')
    #
    inpData['day_count'] = inpData.index[::-1]
    inpData = inpData.sort_values(by='date')
    inpData['DateTime'] = pd.to_datetime(inpData['date'], format="%m/%d/%y")
    inpData = inpData.sort_values(by='DateTime')
    List_Index_NAN_hosptoal = [i[0] for i in list(map(tuple, np.argwhere(~np.isnan(inpData['hospitalizedCurrently'].values))))]
    indFirstValidRow = min(List_Index_NAN_hosptoal)
    indLastValidRow = max(List_Index_NAN_hosptoal)
    inpData = inpData.iloc[indFirstValidRow:indLastValidRow]
    #
    # adding I and dI from data
    inpData['I_data'] = inpData['hospitalizedCurrently'].divide(I2HospitalRate).apply(np.round)
    inpData['I_data'] = inpData['I_data'].shift(periods=-1 * delay_from_hospital_to_I)
    inpData['dI_data'] = inpData['I_data'].diff()
    #
    inpData = inpData.iloc[0:-delay_from_hospital_to_I]
    #
    list_sorted_Dates = sorted(inpData['DateTime'].tolist())
    #
    # adding I and dI from smoothing of the I_date and differentiating I_data
    inpData['I'] = scipy.signal.savgol_filter(inpData['I_data'], 51, 3)  # window size 51, polynomial order 3
    inpData['I'] = inpData['I'].apply(np.floor)
    inpData['dI'] = inpData['I'].diff()
    #
    # adding E and dE
    inpData['E'] = (inpData['dI'] + I2Rrate * inpData['I']).divide(E2Irate)
    inpData['dE'] = inpData['E'].diff()
    #
    dict_Dates_to_index = dict()
    dict_index_to_Dates = dict()
    for j in range(0, len(inpData.index)):
        dict_Dates_to_index[inpData['DateTime'].iloc[j]] = j
        dict_index_to_Dates[str(j)] = inpData['DateTime'].iloc[j]
    #
    # Adding the terms Beta_I_S, dS, and S
    inpData['Beta_I_S'] = inpData['dE'] + E2Irate * inpData['E']
    inpData['dS'] = -1 * inpData['Beta_I_S']
    #
    inpData['S'] = np.nan
    startIntegrating = False
    for currDateTime in list_sorted_Dates:
        currIndex = dict_Dates_to_index[currDateTime]
        #
        if startIntegrating:
            inpData['S'].iloc[currIndex] = inpData['S'].iloc[currIndex - 1] + inpData['dS'].iloc[currIndex]
        #
        if not startIntegrating:
            if np.isnan(inpData['dS'].iloc[currIndex]) and (not np.isnan(inpData['dS'].iloc[currIndex + 1])):
                inpData['S'].iloc[currIndex] = S0
                startIntegrating = True
    #
    # adding Beta
    inpData['Beta_not_smooth'] = np.nan
    inpData['Beta_not_smooth'].iloc[2:] = N0 * inpData['Beta_I_S'].iloc[2:] / (inpData['I'].iloc[2:] * inpData['S'].iloc[2:])
    inpData['Beta'] = np.nan
    inpData['Beta'].iloc[2:] = scipy.signal.savgol_filter(inpData['Beta_not_smooth'].iloc[2:], 17, 3)
    #
    # adding d Beta
    inpData['dBeta'] = inpData['Beta'].diff()
    #
    #
    # adding Beta_coeff including Beta of summer and winter (this part should be modified for each state)
    inpData["DateMonthYearString"] = inpData["DateTime"].astype(str)
    def calc_Beta_coeff(inpData):
        curr_Beta = Beta_coeff_summer * (1 + (Beta_coeff_winter - Beta_coeff_summer) * np.cos(
            math.pi * (inpData['day_count'] - 320) / (2 * 365)))
        # return curr_Beta
        if inpData["DateMonthYearString"] < '2020-05-01':
            curr_Beta = 0.5 * curr_Beta  # first lockdown
        elif (inpData["DateMonthYearString"] >= '2020-05-01') and (inpData["DateMonthYearString"] < '2020-06-15'):
            curr_Beta = curr_Beta  # re-opening
        elif (inpData["DateMonthYearString"] >= '2020-06-15') and (inpData["DateMonthYearString"] < '2020-07-01'):
            curr_Beta = 0.6 * curr_Beta  # partial lockdown
        elif (inpData["DateMonthYearString"] >= '2020-07-01') and (inpData["DateMonthYearString"] >= '2020-07-01'):
            curr_Beta = 0.8 * curr_Beta  # mask mandate
        else:
            curr_Beta = curr_Beta  #
        return curr_Beta
        #
        # if inpData["I"] >= 0.008:
        #     curr_Beta = 0.6 * curr_Beta
        # else:
        #     curr_Beta = curr_Beta
        # return curr_Beta
        #
        # if inpData["hospitalizedCurrently"] >= 0.9*13846:
        #     curr_Beta = 0.7 * curr_Beta
        # else:
        #     curr_Beta = curr_Beta
        # return curr_Beta

    #
    inpData['Beta_coeff'] = inpData.apply(calc_Beta_coeff, axis=1)
    #
    # adding d x
    inpData['dx_not_smooth'] = np.nan
    inpData['dx_not_smooth'].iloc[3:] = -1 * (1 / inpData['Beta_coeff'].iloc[3:]) * inpData['dBeta'].iloc[3:]
    inpData['dx'] = np.nan
    inpData['dx'].iloc[3:] = scipy.signal.savgol_filter(inpData['dx_not_smooth'].iloc[3:], 27, 3)
    #
    # Adding x
    inpData['x'] = 1 - (1 / inpData['Beta_coeff']) * inpData['Beta']
    #
    # calculating dn / dt
    inpData['dn'] = (-1) * epsil * inpData['dI'] / N0
    #
    # Adding n
    inpData['n'] = n0 - epsil * inpData['I'] / N0
    #
    ## saving the results
    inpData = inpData.reindex(columns=
                              ['date',
                               'day_count',
                               'DateTime',
                               'state',
                               'S',
                               'dS',
                               'E',
                               'dE',
                               'I',
                               'dI',
                               'x',
                               'dx',
                               'n',
                               'dn',
                               'Beta',
                               'dBeta',
                               'Beta_coeff',
                               'dx_not_smooth',
                               'I_data',
                               'dI_data',
                               'Beta_I_S',
                               'death_smooth',
                               'd_death',
                               'Beta_not_smooth',
                               'Beta_smooth',
                               'death',
                               'deathIncrease',
                               'hospitalizedCurrently',
                               'inIcuCurrently',
                               'positive',
                               'positiveCasesViral',
                               'positiveIncrease',
                               'positiveTestsAntibody',
                               'positiveTestsAntigen',
                               'positiveTestsViral',
                               'recovered',
                               'recoverdIncremental',
                               'totalTestResults',
                               'totalTestResultsIncrease',
                               'totalTestsAntibody',
                               'totalTestsAntigen',
                               'totalTestsViral',
                               'totalTestsViralIncrease'
                               ])
    #
    inpData.to_csv('./Data/NEW-' + state_name + '-history-Final-Data-with-seasonality-and-sinusoid.csv')
    #
    return None


