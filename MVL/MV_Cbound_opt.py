"""
This code is implements the C-bound optimization problem for the Multiview learning algorithm PB-MVBoost.

Related Paper:
Multiview Boosting by Controlling the Diversity and the Accuracy of View-specific Voters
by Anil Goyal, Emilie Morvant, Pascal Germain and Massih-Reza Amini

Link to the paper:
https://arxiv.org/abs/1808.05784
"""
__author__="Anil Goyal"
import numpy as np
from scipy.optimize import minimize

class MV_Cbount_opt(object):
    """
    This class solves the C-bound optimization problem for the Multiview learning algorithm PB-MVBoost.
    It learns the weights for over the views for our algorithm.
    """
    def __init__(self,initial_guess,risk_vector,disagreement_vector):
        """

        :param initial_guess: vector for the initial guess of weights
        :param risk_vector: Risk vector
        :param disagreement_vector: Vector for disagreement values
        """

        self.initial_guess=initial_guess
        self.risk_vector=risk_vector
        self.disagreement_vector=disagreement_vector



    def func(self, x, r,d,sign=1):
        """ Objective function """
        num=1-2 * (sum(x*r))
        den=1-2 * (sum(x*d))

        return sign * ((num)**2 / den)

    def func_deriv(self, x, r,d, sign=1):
        """ Derivative of objective function """
        num = 1 - 2 * (sum(x*r))
        den = 1 - 2 * (sum(x*d))

        dfdx= sign * ((-1 * 4 * r * num * den + 2 * d * (num)**2) / (den ** 2))

        return np.array(dfdx)


    def learn_weights(self):
        """
        Learns weights
        :param self:
        :return:
        """
        x = self.initial_guess
        r = self.risk_vector
        d = self.disagreement_vector
        arguments = (r, d, -1)




        cons = ({'type': 'eq',
                        'fun': lambda x: np.array([sum(x) - 1]),
                         'jac': lambda x: np.array(x)}
                        )

        res = minimize(self.func, x, args=arguments, bounds=tuple((0,None) for i in range(len(x))), jac=self.func_deriv,
                               constraints=cons, method='SLSQP', options={'disp': False})


        if np.isnan(res.x[0]):
            return self.initial_guess
        else:
            return res.x



