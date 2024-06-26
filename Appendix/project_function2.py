from services.strategies import *



def project_function(periodReturns, periodFactRet, x0, strategy):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :param x0: initial portfolio weights
    :param strategy: strategy name as string
    :return: the allocation as a vector
    """
    if strategy == 'BSS_RiskParity':
        Strategy = BSS_RiskParity()
    elif strategy == 'OLS_RiskParity':
        Strategy = OLS_RiskParity()
    elif strategy == 'FF_RiskParity':
        Strategy = FF_RiskParity()
    elif strategy == 'LASSO_RiskParity':
        Strategy = LASSO_RiskParity()        
    elif strategy == 'LASSO_MVO':
        Strategy = LASSO_MVO()
    elif strategy == 'BSS_MVO':
        Strategy = BSS_MVO()
    elif strategy == 'OLS_MVO':
        Strategy = OLS_MVO()
    elif strategy == 'FF_MVO':
        Strategy = FF_MVO()
    elif strategy == 'LASSO_robustMVO':
        Strategy = LASSO_robustMVO()
    elif strategy == 'BSS_robustMVO':
        Strategy = BSS_robustMVO()
    elif strategy == 'OLS_robustMVO':
        Strategy = OLS_robustMVO()
    elif strategy == 'FF_robustMVO':
        Strategy = FF_robustMVO()
    elif strategy == 'LASSO_CVaR':
        Strategy = LASSO_CVaR()
    elif strategy == 'BSS_CVaR':
        Strategy = BSS_CVaR()
    elif strategy == 'OLS_CVaR':
        Strategy = OLS_CVaR()
    elif strategy == 'FF_CVaR':
        Strategy = FF_CVaR()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    x = Strategy.execute_strategy(periodReturns, periodFactRet)
    return x