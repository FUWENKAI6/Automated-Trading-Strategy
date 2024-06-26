import cvxpy as cp
import numpy as np
from scipy.stats import chi2
import numpy as np
import cvxpy as cp

def risk_parity(Q, kappa=0.5):
    n = Q.shape[0]
    x = cp.Variable(n)
    
    # Objective function: 0.5 * x.T @ Q @ x - kappa * sum(log(x))
    portfolio_variance = 0.5 * cp.quad_form(x, Q)
    log_barrier = -kappa * cp.sum(cp.log(x))
    objective = cp.Minimize(portfolio_variance + log_barrier)
    
    # Constraints
    constraints = [
        x >= 0,  # No short selling
    ]
    
    # Problem definition
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve(solver=cp.ECOS)
    
    x.value /= np.sum(x.value)  # Normalize weights to sum to 1
    sigma_p = np.sqrt(x.value.T @ Q @ x.value)
    mrc = (Q @ x.value) / sigma_p
    risk_contribution = x.value * mrc
    return x.value, risk_contribution

def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value
def robustMVO(mu, Q, T, lambda_ , alpha ):
    """
    Construct a robust MVO portfolio considering uncertainties in mu and Q.
    Args:
        mu (np.ndarray): Expected returns.
        Q (np.ndarray): Covariance matrix.
        lambda_ (float): Risk aversion coefficient.
        alpha (float): Confidence level.
        T (int): Number of return observations.
    Returns:
        np.ndarray: Optimal portfolio weights.
    """
    # Number of assets
    n = len(mu)
    
    # Radius of the uncertainty set
    ep = np.sqrt(chi2.ppf(alpha, n))
    
    # Squared standard error of expected returns
    Theta = np.diag(np.diag(Q)) / T
    
    # Square root of Theta
    sqrtTh = np.sqrt(Theta)
    
    # Initial portfolio (equally weighted)
    x0 = np.ones(n) / n
    
    # Variables
    x = cp.Variable(n)
    
    # Objective function
    objective = cp.Minimize(lambda_ * cp.quad_form(x, Q) - mu.T @ x + ep * cp.norm(sqrtTh @ x))
    
    # Constraints
    constraints = [
        cp.sum(x) == 1,  # Sum of weights equals 1
        x >= 0           # No short sales
    ]
    
    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return x.value
def CVaR(mu, returns, alpha=0.95):
    returns = np.array(returns)  # Ensure returns is a numpy array
    s, n = returns.shape  # s: number of scenarios, n: number of assets

    # Ensure mu is a column vector
    mu = mu.flatten()  # Ensure mu is a 1D array
    target_return = 1.1 * np.mean(mu)  # Set the target return

    # Define variables
    x = cp.Variable(n)  # Portfolio weights
    z = cp.Variable(s)  # Auxiliary variables
    gamma = cp.Variable()  # Value at Risk

    # Define constraints
    constraints = [
        z >= 0,  # Auxiliary variables must be non-negative
        z >= -returns @ x - gamma,  # CVaR constraint
        cp.sum(x) == 1,  # Sum of weights equals 1 (no leverage)
        mu @ x >= target_return,  # Ensure target return is met
        x >= 0  # No short selling
    ]


    # Define the objective function
    k = 1 / ((1 - alpha) * s)
    objective = cp.Minimize(gamma + k * cp.sum(z))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    # Return the optimal portfolio weights
    return x.value
