# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Functions to generate loss symbols for sequence-to-sequence models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import mxnet as mx
from mxnet.metric import EvalMetric

from . import config
from . import constants as C
from . import kl_divergence as kl

logger = logging.getLogger(__name__)


def label_mask(values: mx.sym.Symbol, label: mx.sym.Symbol, ignore_symbol: mx.sym.Symbol) -> mx.sym.Symbol:
    mask = label != ignore_symbol
    return values * mask


class LossConfig(config.Config):
    """
    Loss configuration.

    :param name: Loss name.
    :param vocab_size: Target vocab size.
    :param normalization_type: How to normalize the loss.
    :param label_smoothing: Optional smoothing constant for label smoothing.
    """

    def __init__(self,
                 name: str,
                 vocab_size: int,
                 normalization_type: str,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.normalization_type = normalization_type
        self.label_smoothing = label_smoothing


def get_loss(loss_config: LossConfig) -> 'Loss':
    """
    Returns Loss instance.

    :param loss_config: Loss configuration.
    """
    if loss_config.name == C.CROSS_ENTROPY:
        return CrossEntropyLoss(loss_config)
    else:
        raise ValueError("unknown loss name: %s" % loss_config.name)


class Loss(ABC):
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol and the softmax outputs.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric(self) -> EvalMetric:
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        pass


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: CrossEntropy(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        self.loss_config = loss_config

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbol.
        """
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            normalization = "valid"
        elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
            normalization = "null"
        else:
            raise ValueError("Unknown loss normalization type: %s" % self.loss_config.normalization_type)
        return [mx.sym.SoftmaxOutput(data=logits,
                                     label=labels,
                                     ignore_label=C.PAD_ID,
                                     use_ignore=True,
                                     normalization=normalization,
                                     smooth_alpha=self.loss_config.label_smoothing,
                                     name=C.SOFTMAX_NAME)]

    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)


class KLLoss(ABC):
    """
    A loss for the KL divergence of a distribution p from a distribution q: KL(q||p).

    :param name: The name for this loss.
    :param distribution: Name of the distribution for which to compute the KL divergence.
    """

    def __init__(self, name: str, distribution_name: str) -> List[mx.sym.Symbol]:
        self.kl_divergence = kl.get_kl_divergence(distribution_name)
        self.name = name

    def get_loss(self, params_q: List[mx.sym.Symbol], params_p: List[mx.sym.Symbol], labels: Optional[mx.sym.Symbol] = None, annealing_factor: Optional[mx.sym.Symbol] = None) -> List[mx.sym.Symbol]:
        """
        Returns the loss symbol for the KL divergence.

        :param params_q: The parameters of the distribution q.
        :param params_p: The parameters of the distribution p.
        :param labels: Optionally supply labels. If the labels is equal to the PAD_ID the kl loss will ingnore
        the corresponding term.
        :return: A loss symbol wrapped in a list.
        """
        kl_values = self.kl_divergence(*params_p, *params_q)
        if labels is not None:
            kl_values = label_mask(kl_values, labels, C.PAD_ID)
        if annealing_factor is not None:
            kl_values = kl_values * annealing_factor
        return [mx.sym.MakeLoss(data=kl_values, name=self.name)]


class CrossEntropyMetric(EvalMetric):
    """
    Version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.CROSS_ENTROPY,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config

    def cross_entropy(self, pred, label, ignore):
        prob = mx.nd.pick(pred, label.astype(dtype="int32"))
        prob = prob * (1 - ignore) + ignore
        loss = -mx.nd.log(prob + 1e-8)  # pylint: disable=invalid-unary-operand-type
        return loss

    def cross_entropy_smoothed(self, pred, label, ignore):
        label_dist = mx.nd.one_hot(indices=label.astype(dtype='int32'),
                                   depth=self.loss_config.vocab_size,
                                   on_value=1.0 - self.loss_config.label_smoothing,
                                   off_value=self.loss_config.label_smoothing /
                                             (self.loss_config.vocab_size - 1.0))
        label_dist = mx.nd.where(1 - ignore, label_dist, mx.nd.zeros_like(label_dist))
        loss = label_dist * (- mx.nd.log(pred + 1e-8))  # pylint: disable=invalid-unary-operand-type
        return loss

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            batch_size = label.shape[0]
            label = label.as_in_context(pred.context).reshape((label.size,))
            # Ignore padding
            # TODO: contribute ignoring padding for cross-entropy back to MXNet
            ignore = (label == C.PAD_ID).astype(dtype=pred.dtype)

            if self.loss_config.label_smoothing > 0.0:
                loss = self.cross_entropy_smoothed(pred, label, ignore)
            else:
                loss = self.cross_entropy(pred, label, ignore)

            # Sum, normalizing if needed
            if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
                loss = loss / mx.nd.sum(1 - ignore)
                self.num_inst += 1
            elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size
            self.sum_metric += mx.nd.sum(loss).asscalar()


class CrossEntropyNegElboMetric(CrossEntropyMetric):
    """
    Elbo with version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def update(self, labels, preds):
        super().update(labels, preds)
        likelihoods, kl_values , *_ = preds
        self.sum_metric += mx.nd.sum(kl_values).asscalar()


class CrossEntropyElboMetric(CrossEntropyMetric):
    """
    Elbo with version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.ELBO,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None,
                 annealing_factor: Optional[float] = 1.0,
                 annealing_increase: [Optional] = 0.0,
                 kl_terms: Optional[int] = 1) -> None:
        super().__init__(loss_config, name, output_names, label_names)
        self.annealing_factor = annealing_factor
        self.annealing_increase = annealing_increase
        self.kl_terms = kl_terms

    def update(self, labels, preds):
        likelihoods, *kl_values = preds
        for label, lik in zip(labels, [likelihoods]):
            batch_size = label.shape[0]
            label = label.as_in_context(lik.context).reshape((label.size,))
            # Ignore padding
            # TODO: contribute ignoring padding for cross-entropy back to MXNet
            ignore = (label == C.PAD_ID).astype(dtype=lik.dtype)

            if self.loss_config.label_smoothing > 0.0:
                loss = self.cross_entropy_smoothed(lik, label, ignore)
            else:
                loss = self.cross_entropy(lik, label, ignore)

            # Sum, normalizing if needed
            if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
                loss = loss / mx.nd.sum(1 - ignore)
                self.num_inst += 1
            elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size

            kl_sum = 0
            for i in range(self.kl_terms):
                kl_term = kl_values[i]
                if kl_term.shape[0] == likelihoods.shape[0]:
                    if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
                        kl_term = kl_term / mx.nd.sum(1 - ignore)
                    kl_term = label_mask(kl_term, label, C.PAD_ID)
                kl_sum += mx.nd.sum(kl_term)

            self.sum_metric -= mx.nd.sum(loss).asscalar() + self.annealing_factor * kl_sum.asscalar()
            self.annealing_factor += self.annealing_increase
            if self.annealing_factor > 1.0:
                self.annealing_factor = 1.0
                self.annealing_increase = 0.0


class KlMetric(EvalMetric):

    def __init__(self,
                 name: str = "kl_divergence",
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None,
                 annealing_factor: Optional[float] = 1.0,
                 annealing_increase: Optional[float] = 0.0,
                 kl_terms: Optional[int] = 1):
        super().__init__(name, output_names, label_names)
        self.annealing_factor = annealing_factor
        self.annealing_increase = annealing_increase
        self.kl_terms = kl_terms

    def update(self, labels, preds):
        likelihoods, *kl_values = preds
        label = labels[0]
        batch_size = label.shape[0]
        label = label.as_in_context(likelihoods.context).reshape((label.size,))

        kl_sum = 0
        for i in range(self.kl_terms):
            kl_term = kl_values[i]
            if kl_term.shape[0] == likelihoods.shape[0]:
                kl_term = label_mask(kl_term, label, C.PAD_ID)
            kl_sum += mx.nd.sum(kl_term)

        self.num_inst += batch_size
        self.sum_metric += self.annealing_factor * kl_sum.asscalar()
        self.annealing_factor += self.annealing_increase
        if self.annealing_factor > 1.0:
            self.annealing_factor = 1.0
            self.annealing_increase = 0.0
