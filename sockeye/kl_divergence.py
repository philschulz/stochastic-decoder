from typing import Optional, Callable
from . import constants as C

import mxnet as mx


def get_kl_divergence(distribution_name: str) -> Callable:
    if distribution_name == C.DIAGONAL_GAUSS:
        return diagonal_gaussian_kl
    else:
        raise ValueError("Unsupported distribution")


def diagonal_gaussian_kl(mean_q: mx.sym.Symbol, std_q: mx.sym.Symbol, mean_p: Optional[mx.sym.Symbol] = None,
                         std_p: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
    """
    Computes the KL divergence KL(q||p) of a Gaussian distribution p from a Gaussian distribution q. Both Gaussians
    are assumed to have diagonal covariance matrices. If the parameters of of p are not provided, it is assumed
    to be the standard normal distribution.

    :param mean_q: The mean of q. Shape: (batch_size, dim)
    :param std_q: The square roots the variances of q. Shape: (batch_size, dim)
    :param mean_p: The mean of p. Shape: (batch_size, dim)
    :param std_p: The square roots of the variances of p. Shape: (batch_size, dim)
    :return: The KL divergence of p from q. Shape: (batch_size, dim)
    """
    if mean_p is None or std_p is None:
        return mx.sym.sum(0.5 * (std_q ** 2 + mean_q ** 2 - 2 * mx.sym.log(std_q) - 1), axis=1)
    else:
        std_ratio = std_q / std_p
        return mx.sym.sum(0.5 * (std_ratio ** 2 + ((mean_q - mean_p) / std_p) ** 2 - 2 * mx.sym.log(std_ratio) - 1),
                          axis=1)
