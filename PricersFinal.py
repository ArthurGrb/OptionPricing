import numpy as np
from scipy.stats import norm, mvn
import pandas as pd
from math import isnan


def BlackScholes(S, K, T, r, sigma, div, opt):
    """
    Returns option price using Black-Scholes model

    :param S: Spot price (0.0-30000.0)
    :param K: Strike price (0.0-30000.0)
    :param T: Time to expiry (0.0-5.0)
    :param r: Risk-free rate (0.0-0.2)
    :param sigma: Volatility (0.0-1.5)
    :param Opt: Option type ('c' or 'p')
    :return: Option price (0.0-30000.0)
    """
    d1 = (np.log(S / K) + (r - div + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pv = np.where(opt == 'c',
                  S * np.exp(-div * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
                  K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-div * T) * norm.cdf(-d1))
    return pv

# TODO: maybe this is class
def BjerksundStensland(S, K, T, r, sigma, div, opt):
    """
    Returns option price using Bjerksund-Stensland model

    :param S: Spot price (0.0-30000.0)
    :param K: Strike price (0.0-30000.0)
    :param T: Time to expiry (0.0-5.0)
    :param r: Risk-free rate (0.0-0.2)
    :param sigma: Volatility (0.0-1.5)
    :param div: Dividend yield (0.0-1.5)
    :param Opt: Option type ('c' or 'p')
    :return: Option price (0.0-30000.0)
    """
    S_df = S
    K_df = K
    T_df = T
    r_df = r
    sigma_df = sigma
    div_df = div
    opt_df = opt
    BjerksundValue = pd.DataFrame(pd.np.empty((S_df.shape[0], 1)))
    for i in range(S_df.shape[0]):
        S = S_df[i]
        K = K_df[i]
        T = T_df[i]
        r = r_df[i]
        sigma = sigma_df[i]
        div = div_df[i]
        opt = opt_df[i]

        b = r - div

        if opt == 'p':
            S, K = K, S
            div = r
            r = r - b
            b = r - div

        if b >= r:
            pv =  BlackScholes(S, K, T, r, sigma, div, opt)

        else:
            beta = (1 / 2) - (b / (sigma ** 2)) + np.sqrt((b / (sigma ** 2) - (1 / 2)) ** 2 + 2 * r / sigma ** 2)

            def alpha(X):
                return (X - K) * X ** -beta

            B0 = np.maximum(K, (r / (r - b)) * K)
            BInf = (beta / (beta - 1)) * K

            def h(T):
                return -(b * T + 2 * sigma * np.sqrt(T)) * (K ** 2 / ((BInf - B0) * B0))

            def X(T):
                return B0 + (BInf - B0) * (1 - np.exp(h(T)))

            def phi(S, T, gamma, H, X):
                d1 = -(np.log(S / H) + (b + (gamma - 0.5) * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
                d2 = -(np.log(X ** 2 / (S * H)) + (b + (gamma - 0.5) * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
                lamb = -r + gamma * b + 0.5 * gamma * (gamma - 1) * (sigma ** 2)
                kappa = (2 * b) / (sigma ** 2) + (2 * gamma - 1)
                return np.exp(lamb * T) * (S ** gamma) * (norm.cdf(d1) - ((X / S) ** kappa) * norm.cdf(d2))

            # Taken from dedwards25/Python_Option_Pricing
            def _cbnd(a, b, rho):
                # This distribution uses the Genz multi-variate normal distribution
                # code found as part of the standard SciPy distribution
                lower = np.array([0, 0])
                upper = np.array([a, b])
                infin = np.array([0, 0])
                correl = rho
                error, value, inform = mvn.mvndst(lower, upper, infin, correl)
                return value

            def psi(S, big_T, gamma, H, big_X, small_X, small_T):
                lamb = -r + gamma * b + 0.5 * gamma * (gamma - 1) * (sigma ** 2)
                kappa = 2 * b / (sigma ** 2) + (2 * gamma - 1)

                small_d1 = -(np.log(S / small_X) + (b + (gamma - 0.5) * (sigma ** 2)) * small_T) / (sigma * np.sqrt(small_T))
                small_d2 = -(np.log((big_X ** 2) / (S * small_X)) + (b + (gamma - 0.5) * (sigma ** 2)) * small_T) / (sigma * np.sqrt(small_T))
                small_d3 = -(np.log(S / small_X) - (b + (gamma - 0.5) * (sigma ** 2)) * small_T) / (sigma * np.sqrt(small_T))
                small_d4 = -(np.log((big_X ** 2) / (S * small_X)) - (b + (gamma - 0.5) * (sigma ** 2)) * small_T) / (sigma * np.sqrt(small_T))
                big_d1   = -(np.log(S / H) + (b + (gamma - 0.5) * (sigma ** 2)) * big_T) / (sigma * np.sqrt(big_T))
                big_d2   = -(np.log((big_X ** 2) / (S * H)) + (b + (gamma - 0.5) * (sigma ** 2)) * big_T) / (sigma * np.sqrt(big_T))
                big_d3   = -(np.log((small_X ** 2) / (S * H)) + (b + (gamma - 0.5) * (sigma ** 2)) * big_T) / (sigma * np.sqrt(big_T))
                big_d4   = -(np.log((S * (small_X ** 2) / (H * (big_X ** 2)))) + (b + (gamma - 0.5) * (sigma ** 2)) * big_T) / (sigma * np.sqrt(big_T))

                return np.exp(lamb * big_T) * (S ** gamma) * (_cbnd(small_d1, big_d1, np.sqrt(small_T / big_T))
                                                              - (((big_X / S) ** kappa) * _cbnd(small_d2, big_d2, np.sqrt(small_T / big_T)))
                                                              - ((small_X / S) ** kappa) * _cbnd(small_d3, big_d3, -np.sqrt(small_T / big_T))
                                                              + ((small_X / big_X) ** kappa) * _cbnd(small_d4, big_d4, -np.sqrt(small_T / big_T)))

            big_T = T
            small_T = 0.5 * (np.sqrt(5) - 1) * big_T
            big_X = X(big_T)
            small_X = X(big_T - small_T)

            if S >= big_X:
                pv = S - K

            else:
                pv = alpha(big_X) * (S ** beta) \
                     - alpha(big_X) * phi(S, small_T, beta, big_X, big_X) \
                     + phi(S, small_T, 1.0, big_X, big_X) \
                     - phi(S, small_T, 1.0, small_X, big_X) \
                     - K * phi(S, small_T, 0.0, big_X, big_X) \
                     + K * phi(S, small_T, 0.0, small_X, big_X) \
                     + alpha(small_X) * phi(S, small_T, beta, small_X, big_X) \
                     - alpha(small_X) * psi(S, big_T, beta, small_X, big_X, small_X, small_T) \
                     + psi(S, big_T, 1.0, small_X, big_X, small_X, small_T) \
                     - psi(S, T, 1.0, K, big_X, small_X, small_T) \
                     - K * psi(S, big_T, 0.0, small_X, big_X, small_X, small_T) \
                     + K * psi(S, big_T, 0.0, K, big_X, small_X, small_T)

        if isnan(pv):
            pv = 0.0

        blsprice = BlackScholes(S, K, T, r, sigma, div, opt)
        if pv < blsprice:
            pv = blsprice

        BjerksundValue.loc[i] = pv


    return BjerksundValue

def BinomialTree(S, K, T, r, sigma, div, opt, N):
    """
    Returns option price using Binomial model and Broaie-Dutemple convergence

    :param S: Spot price (0.0-30000.0)
    :param K: Strike price (0.0-30000.0)
    :param T: Time to expiry (0.0-5.0)
    :param r: Risk-free rate (0.0-0.2)
    :param sigma: Volatility (0.0-1.5)
    :param Opt: Option type ('c' or 'p')
    :param N: Number of steps in tree (0-3000)
    :return: Option price (0.0-30000.0)
    """
    BinomialValue = pd.DataFrame(pd.np.zeros((S.shape[0], 1)))
    for h in range(S.shape[0]):
        At = T[h] / N
        u = np.exp((sigma[h]) * np.sqrt(At))
        d = 1. / u
        p = (np.exp((r[h] - div[h]) * At) - d) / (u - d)

        # Binomial price tree
        stockvalue = np.zeros((N + 1, N + 1))
        stockvalue[0, 0] = S[h]
        for i in range(1, N + 1):
            stockvalue[i, 0] = stockvalue[i - 1, 0] * u
            for j in range(1, i + 1):
                stockvalue[i, j] = stockvalue[i - 1, j - 1] * d

        # option value at final node
        optionvalue = np.zeros((N + 1, N + 1))
        for j in range(N + 1):
            if opt[h] == "c":  # Call
                optionvalue[N, j] = max(0, stockvalue[N, j] - K[h])
            elif opt[h] == "p":  # Put
                optionvalue[N, j] = max(0, K[h] - stockvalue[N  , j])

        # backward calculation for option price
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                if opt[h] == "p":
                    optionvalue[i, j] = max(0, K[h] - stockvalue[i, j], np.exp(-r[h] * At) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))
                elif opt[h] == "c":
                    optionvalue[i, j] = max(0, stockvalue[i, j] - K[h], np.exp(-r[h] * At) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))

        blsprice = BlackScholes(S[h], K[h], T[h], r[h], sigma[h], div[h], opt[h])
        if optionvalue[0, 0] < blsprice:
            optionvalue[0, 0] = blsprice
        BinomialValue.loc[h] = optionvalue[0, 0]
        if h % 1000 == 0:
            print(h)
    return BinomialValue


def NeuralNetwork(S, K, T, r, sigma, Opt, model):
    """
    Returns option price using Neural network

    :param S: Spot price (0.0-30000.0)
    :param K: Strike price (0.0-30000.0)
    :param T: Time to expiry (0.0-5.0)
    :param r: Risk-free rate (0.0-0.2)
    :param sigma: Volatility (0.0-1.5)
    :param Opt: Option type ('c' or 'p')
    :param model: Neural network model
    :return: Option price (0.0-30000.0)
    """


