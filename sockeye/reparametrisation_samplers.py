from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import mxnet as mx

from sockeye import constants as C
from sockeye.config import Config


class ReparametrisationSamplerConfig(Config):
    """
    Configuration of a stochastic Encoder.

    :param distribution: The distribution of the latent variable.
    :param latent_dim: The dimensionality of the latent variable.
    Alternatively, the last encoder state is used for that purpose.
    """

    def __init__(self, distribution_name: str, latent_dim: int) -> None:
        super().__init__()
        self.distribution_name = distribution_name
        self.latent_dim = latent_dim


class ReparametrisationSampler(ABC):
    """
    An encoder that enodes the input as distribution.

    :param latent_dim: dimensionality of the latent variable.
    :param prefix: Name prefix for symbols of this encoder.
    """

    def __init__(self, latent_dim: int,
                 prefix: str = C.SAMPLER_PREFIX):
        self.latent_dim = latent_dim
        self.prefix = prefix

    @abstractmethod
    def sample(self, data: mx.sym.Symbol, deterministic: bool = False) -> Tuple[mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Encode data in a distribution and sample from that distribution.

        :param data: Input data.
        :param deterministic: Samples are computed deterministically from the paramters. Should not be used during
        training, only for translation.
        :return: A sample of the latent variable and the grouped parameters of the distribution.
        """
        raise NotImplementedError()


def get_reparametrisation_sampler(config: ReparametrisationSamplerConfig, prefix: str = "") -> ReparametrisationSampler:
    """
    Produces a stochastic encoder.

    :param config: Configuration of the stochastic encoder.
    :prefix prefix: Prefix for this sampler.
    :return: A stochastic encoder.
    """

    if config.distribution_name == C.DIAGONAL_GAUSS:
        return DiagonalGaussianSampler(config.latent_dim, prefix)
    elif config.distribution_name == C.FULL_RANK_GAUSS:
        return FullRankGaussianSampler(config.latent_dim, prefix)
    else:
        raise ValueError("Unsupported stochastic encoder configuration")


class DiagonalGaussianSampler(ReparametrisationSampler):
    """
    An encoder infers the parameters of a diagonal Gaussian distribution and samples a value from it.

    :param latent_dim: dimensionality of the latent variable.
    :param prefix: Name prefix for symbols of this encoder.
    """

    def __init__(self,
                 latent_dim: int,
                 prefix=C.GAUSS_PREFIX + C.SAMPLER_PREFIX) -> None:
        super().__init__(latent_dim, prefix)
        self.mean_l1_w = mx.sym.Variable("%smean_layer1_weight" % self.prefix)
        self.mean_l1_b = mx.sym.Variable("%smean_layer1_bias" % self.prefix)
        self.mean_l2_w = mx.sym.Variable("%smean_layer2_weight" % self.prefix)
        self.mean_l2_b = mx.sym.Variable("%smean_layer2_bias" % self.prefix)
        self.scale_l1_w = mx.sym.Variable("%sscale_layer1_weight" % self.prefix)
        self.scale_l1_b = mx.sym.Variable("%sscale_layer1_bias" % self.prefix)
        self.scale_l2_w = mx.sym.Variable("%sscale_layer2_weight" % self.prefix)
        self.scale_l2_b = mx.sym.Variable("%sscale_layer2_bias" % self.prefix)

    def sample(self, data: mx.sym.Symbol, deterministic: bool = False) -> Tuple[mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Encodes data in a Gaussian distribution given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param deterministic: Deterministically return mean value. Should not be used during training,
        only for translation.
        :return: A sample of the latent Gaussian variable and the grouped parameters of the distribution.
        """
        mean = self._compute_mean(data)
        scale = self._compute_scale(data)

        latent_variable = mean if deterministic else self._sample_value(mean, scale)

        return latent_variable, [mean, scale]

    def sample_with_residual_mean(self, data: mx.sym.Symbol, origin: mx.sym.Symbol, deterministic: bool = False) -> \
        Tuple[mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Encodes data in a Gaussian distribution given sequence lengths of individual examples and maximum sequence length.
        The sampler computes a residual and adds it to the origin.

        :param data: Input data.
        :param origin: An origin for the coordinate system of sampled values that is not 0. The mean is computed
        wrt to that origin. This can be used to compute the mean as a sum of origin and a predicted residual.
        :param deterministic: Deterministically return mean value. Should not be used during training,
        only for translation.
        :return: A sample of the latent Gaussian variable and the grouped parameters of the distribution.
        """
        mean = self._compute_mean(data, origin)
        scale = self._compute_scale(data)

        latent_variable = mean if deterministic else self._sample_value(mean, scale)

        return latent_variable, [mean, scale]

    def _sample_value(self, mean: mx.sym.Symbol, scale: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Produce a sample from the inferred Gaussian distribution.

        :param mean: The mean of the Gaussian.
        :param scale: The scale parameter of this Gaussian.
        :return: A random Gaussian vector.
        """
        return mean + scale * mx.sym.random_normal(loc=0, scale=1, shape=(0, self.latent_dim))

    def _compute_mean(self, data: mx.sym.Symbol, origin: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Computes the mean of the Gaussian distribution.

        :param data: The input data.
        :param origin: An origin for the coordinate system of sampled values that is not 0. The mean is computed
        wrt to that origin. This can be used to compute the mean as a sum of origin and a predicted residual.
        :return: A Gaussian mean vector.
        """
        mean = mx.sym.FullyConnected(data=data, num_hidden=self.latent_dim * 2, weight=self.mean_l1_w,
                                     bias=self.mean_l1_b, name="%smean_l1_fc" % self.prefix)
        mean = mx.sym.Activation(data=mean, act_type="tanh")
        mean = mx.sym.FullyConnected(data=mean, num_hidden=self.latent_dim, weight=self.mean_l2_w, bias=self.mean_l2_b,
                                     name="%smean_l2_fc" % self.prefix)
        if origin is not None:
            mean = mean + origin
        return mean

    def _compute_scale(self, data: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Computes the diagonal of the Gaussian covariance matrix.

        :param data: The input data.
        :param dim: The dimensionality of the Gaussian.
        :return: The diagonal of the covariance matrix.
        """
        scale = mx.sym.FullyConnected(data=data, num_hidden=self.latent_dim * 2, weight=self.scale_l1_w,
                                      bias=self.scale_l1_b, name="%sscale_l1_fc" % self.prefix)
        scale = mx.sym.Activation(data=scale, act_type="tanh")
        scale = mx.sym.FullyConnected(data=scale, num_hidden=self.latent_dim, weight=self.scale_l2_w,
                                      bias=self.scale_l2_b, name="%sscale_l2_fc" % self.prefix)
        scale = mx.sym.Activation(data=scale, act_type="softrelu")
        return scale


class FullRankGaussianSampler(DiagonalGaussianSampler):
    """
    An encoder that infers the parameters of a full-rank Gaussian distribution and samples a value from it.

    :param latent_dim: dimensionality of the latent variable.
    :param prefix: Name prefix for symbols of this encoder.
    """

    def __init__(self,
                 latent_dim: int,
                 prefix=C.GAUSS_PREFIX + C.SAMPLER_PREFIX) -> None:
        super().__init__(latent_dim, prefix)

    def _compute_scale(self, data: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Computes a cholesky factor of the covariance matrix.

        :param data: The input data.
        :param dim: The dimensionality of the Gaussian.
        :return: A cholesky factor of the covariance matrix.
        """
        size = self.latent_dim ** 2
        row_lengths = mx.sym.arange(1, self.latent_dim + 1)
        cholesky_vector = mx.sym.FullyConnected(data=data, num_hidden=self.latent_dim * 2, weight=self.scale_l1_w,
                                                bias=self.scale_l1_b, name="%sscale_l1_fc" % self.prefix)
        cholesky_vector = mx.sym.Activation(data=cholesky_vector, act_type="tanh")
        cholesky_vector = mx.sym.FullyConnected(data=cholesky_vector, num_hidden=size, weight=self.scale_l2_w,
                                                bias=self.scale_l2_b, name="%sscale_l2_fc" % self.prefix)

        # cholesky_vector: (latent_dim, latent_dim, batch_size)
        cholesky_factor = mx.sym.reshape(mx.sym.transpose(cholesky_vector),
                                         shape=(self.latent_dim, self.latent_dim, -1),
                                         name="%sreshape_choleksy_vector" % self.prefix)
        # mask rows
        cholesky_factor = mx.sym.SequenceMask(data=cholesky_factor, sequence_length=row_lengths,
                                              use_sequence_length=True, name="%smask_choleksy_rows" % self.prefix)
        # (batch_size, latent_dim, latent_dim)
        cholesky_factor = mx.sym.transpose(data=cholesky_factor, name="%sbatch_cholesky_factor" % self.prefix)
        return cholesky_factor

    def _sample_value(self, mean: mx.sym.Symbol, scale: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Produce a sample from the inferred Gaussian distribution.

        :param mean: The mean of the Gaussian.
        :param scale: The scale parameter of this Gaussian.
        :return: A random Gaussian vector.
        """
        # Ugly hack to do shape inference for the noise
        noise = mx.sym.ones_like(mean) * mx.sym.random_normal(loc=0, scale=1, shape=(0, self.latent_dim))
        # noise: (batch_size, latent_dim, 1)
        noise = mx.sym.expand_dims(noise, axis=2)
        # scaled_noise: (batch_size, latent_dim)
        scaled_noise = mx.sym.reshape(mx.sym.batch_dot(lhs=scale, rhs=noise, name="%sscale_noise" % self.prefix),
                                      shape=(-1, self.latent_dim), name="%reshape_scaled_noise")
        return mean + scaled_noise
