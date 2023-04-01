

import numpy as np
from scipy.integrate import odeint
#
# In this formulation the B0 changes with time

def SEIR(y,t,parameters):
    S, E, I, R, x, n = y
    #
    A_fr,A_fd,I2Rrate,E2Irate,N0,epsil,recovRate, x_dot_intercept, inpData, indLastAvailData, Beta_coeff_manual_setting = parameters
    #
    A_n_00 = ((1-n)*A_fd[0][0] + n*A_fr[0][0])
    A_n_01 = ((1-n)*A_fd[0][1] + n*A_fr[0][1])
    A_n_10 = ((1-n)*A_fd[1][0] + n*A_fr[1][0])
    A_n_11 = ((1-n)*A_fd[1][1] + n*A_fr[1][1])
    #
    A_n = [[A_n_00, A_n_01],
           [A_n_10, A_n_11]]
    #
    avg_payoff_C = np.matmul(np.array([1,0]),np.array(A_n));
    avg_payoff_C = np.matmul(avg_payoff_C, np.array([[x],[1-x]]))
    #
    avg_payoff_D = np.matmul(np.array([0,1]),np.array(A_n));
    avg_payoff_D = np.matmul(avg_payoff_D, np.array([[x],[1-x]]))
    #
    curr_Beta_coeff = inpData['Beta_coeff'].iloc[indLastAvailData + int(t)]
    # curr_Beta_coeff = inpData['Beta_coeff'].iloc[indLastAvailData]
    curr_Beta_coeff = Beta_coeff_manual_setting*curr_Beta_coeff # for (70,110)
    #
    dSdt = - 1 * curr_Beta_coeff * (1 - x) * S * I - 1 * curr_Beta_coeff * (1 - x) * S * I + recovRate * R;
    dEdt =   1 * curr_Beta_coeff * (1 - x) * S * I + 1 * curr_Beta_coeff * (1 - x) * S * I - E2Irate * E;
    dIdt = E2Irate * E - I2Rrate * I;
    dRdt = I2Rrate * I - recovRate * R;
    dxdt = x * (1-x) * (avg_payoff_C - avg_payoff_D) + x_dot_intercept;
    dydt = [
    # %% %% %% %% %% %% %%     d S / d t     %% %% %% %% %%
    dSdt,
    # %% %% %% %% %% %% %%     d E / d t     %% %% %% %% %%
    dEdt,
    # %% %% %% %% %% %% %%     d I / d t     %% %% %% %% %%
    dIdt,
    # %% %% %% %% %% %% %%     d R / d t     %% %% %% %% %%
    dRdt,
    # %% %% %% %% %% %% %%     d x / d t     %% %% %% %% %%
    dxdt,
    # %% %% %% %% %% %% %%     d n / d t     %% %% %% %% %%
    -1*epsil*dIdt
    ]
    return dydt