import numpy as np
from services.estimators import *
from services.optimization import *


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x

class FF_MVO:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q , r= FF(returns, factRet)
        x = MVO(mu, Q)
        return x
class LASSO_MVO:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36, lambda_=0.2, K=None):
        self.NumObs = NumObs  # number of observations to use
        self.lambda_ = lambda_  # Regularization parameter for BSS
        self.K = K  # Number of factors to select

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, r= LASSO(returns, factRet, self.lambda_, self.K)
        x = MVO(mu, Q)
        return x
class BSS_MVO:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36, lambda_=None, K=6):
        self.NumObs = NumObs  # number of observations to use
        self.lambda_ = lambda_  # Regularization parameter for BSS
        self.K = K  # Number of factors to select

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, r = BSS(returns, factRet, self.lambda_, self.K)
        x = MVO(mu, Q)
        return x
    
class OLS_RiskParity:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x, risk_contribution = risk_parity(Q)
        return x
    
class FF_RiskParity:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q , r= FF(returns, factRet)
        x, risk_contribution = risk_parity(Q)
        return x
class LASSO_RiskParity:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36, lambda_=0.2, K=None):
        self.NumObs = NumObs  # number of observations to use
        self.lambda_ = lambda_  # Regularization parameter for BSS
        self.K = K  # Number of factors to select

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, r= LASSO(returns, factRet, self.lambda_, self.K)
        x, risk_contribution = risk_parity(Q)
        return x
class BSS_RiskParity:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36, lambda_=None, K=4):
        self.NumObs = NumObs  # number of observations to use
        self.lambda_ = lambda_  # Regularization parameter for BSS
        self.K = K  # Number of factors to select

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, r = BSS(returns, factRet, self.lambda_, self.K)
        x, risk_contribution = risk_parity(Q)
        return x

class OLS_CVaR:
    """
    Uses historical returns and factor returns to estimate the expected returns using OLS
    and then applies CVaR optimization to find the optimal portfolio weights.
    """

    def __init__(self, NumObs=36, alpha=0.95):
        self.NumObs = NumObs  # Number of observations to use
        self.alpha = alpha  # Confidence level for CVaR

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns: Historical returns for the period
        :param factorReturns: Factor returns for the period
        :return: Optimal portfolio weights
        """
        T, n = periodReturns.shape
        # Get the last NumObs observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        # Calculate expected returns using OLS
        mu, Q = OLS(returns, factRet)
        # Apply CVaR optimization
        x = CVaR(mu, returns, self.alpha)
        return x
    
class FF_CVaR:
    def __init__(self, NumObs=36, alpha=0.95):
        self.NumObs = NumObs
        self.alpha = alpha

    def execute_strategy(self, periodReturns, factorReturns):
        T, n = periodReturns.shape
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, _, _ = FF(returns, factRet)
        optimal_weights = CVaR(mu, returns, self.alpha)
        return optimal_weights

    
class LASSO_CVaR:
    def __init__(self, NumObs=36, alpha=0.95, lambda_=0.1):
        self.NumObs = NumObs
        self.alpha = alpha
        self.lambda_ = lambda_

    def execute_strategy(self, periodReturns, factorReturns):
        T, n = periodReturns.shape
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, _, _ = LASSO(returns, factRet, self.lambda_)
        optimal_weights = CVaR(mu, returns, self.alpha)
        return optimal_weights
    
class BSS_CVaR:
    def __init__(self, NumObs=36, alpha=0.95, K=6):
        self.NumObs = NumObs
        self.alpha = alpha

    def execute_strategy(self, periodReturns, factorReturns):
        T, n = periodReturns.shape
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        # Assuming BSS function is implemented and returns mu, Q
        mu, _, _ = BSS(returns, factRet, lambda_=0, K=6)
        optimal_weights = CVaR(mu, returns, self.alpha)
        return optimal_weights

class OLS_robustMVO:
    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T = self.NumObs
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = robustMVO(mu, Q, T, lambda_=0.75, alpha=0.95)

        return x
    
class FF_robustMVO:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        T = self.NumObs
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q , r= FF(returns, factRet)
        x = robustMVO(mu, Q, T, lambda_=0.02, alpha=0.9)

        return x
    
class LASSO_robustMVO:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36, lambda_=0.2, K=None):
        self.NumObs = NumObs  # number of observations to use
        self.lambda_ = lambda_  # Regularization parameter for BSS
        self.K = K  # Number of factors to select

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        T = self.NumObs
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, r= LASSO(returns, factRet, self.lambda_, self.K)
        x = robustMVO(mu, Q, T, lambda_=0.02, alpha=0.9)
        return x
class BSS_robustMVO:
    """
    Uses historical returns to estimate the covariance matrix and expected return for Risk Parity
    """
    def __init__(self, NumObs=36, lambda_=None, K=6):
        self.NumObs = NumObs  # number of observations to use
        self.lambda_ = lambda_  # Regularization parameter for BSS
        self.K = K  # Number of factors to select

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__
        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        T = self.NumObs
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, r = BSS(returns, factRet, self.lambda_, self.K)
        x = robustMVO(mu, Q, T, lambda_=0.02, alpha=0.9)
        return x


