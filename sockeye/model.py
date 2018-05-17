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

import copy
import logging
import os
from typing import Dict, Optional, Tuple

import mxnet as mx

from sockeye import __version__
from sockeye.config import Config
from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import layers
from . import loss
from . import reparametrisation_samplers as rep
from . import utils

logger = logging.getLogger(__name__)


class ModelConfig(Config):
    """
    ModelConfig defines model parameters defined at training time which are relevant to model inference.
    Add new model parameters here. If you want backwards compatibility for models trained with code that did not
    contain these parameters, provide a reasonable default under default_values.

    :param config_data: Used training data.
    :param max_seq_len_source: Maximum source sequence length to unroll during training.
    :param max_seq_len_target: Maximum target sequence length to unroll during training.
    :param vocab_source_size: Source vocabulary size.
    :param vocab_target_size: Target vocabulary size.
    :param config_embed_source: Embedding config for source.
    :param config_embed_target: Embedding config for target.
    :param config_encoder: Encoder configuration.
    :param config_decoder: Decoder configuration.
    :param config_loss: Loss configuration.
    :param weight_tying: Enables weight tying if True.
    :param weight_tying_type: Determines which weights get tied. Must be set if weight_tying is enabled.
    :param config_latent_encoder: Configuration of a latent variable sampler.
    :param config_inference_encoder: Configuration of an inference net latent variable sampler.
    :param latent_variable_average_states: Average the encoder states before computing the latent distribution.
    :param annealing: Whether to use annealing for the kl loss in variational inference.
    Otherwise take the last encoder state.
    """
    def __init__(self,
                 config_data: data_io.DataConfig,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 vocab_source_size: int,
                 vocab_target_size: int,
                 config_embed_source: Config,
                 config_embed_target: Config,
                 config_encoder: Config,
                 config_decoder: Config,
                 config_loss: loss.LossConfig,
                 weight_tying: bool = False,
                 weight_tying_type: Optional[str] = C.WEIGHT_TYING_TRG_SOFTMAX,
                 weight_normalization: bool = False,
                 config_latent_encoder_sampler: Optional[rep.ReparametrisationSamplerConfig] = None,
                 config_inference_sampler: Optional[rep.ReparametrisationSamplerConfig] = None,
                 latent_variable_average_states: bool = False,
                 annealing: bool = False):
        super().__init__()
        self.config_data = config_data
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size
        self.config_embed_source = config_embed_source
        self.config_embed_target = config_embed_target
        self.config_encoder = config_encoder
        self.config_decoder = config_decoder
        self.config_loss = config_loss
        self.weight_tying = weight_tying
        self.weight_tying_type = weight_tying_type
        self.weight_normalization = weight_normalization
        if weight_tying and weight_tying_type is None:
            raise RuntimeError("weight_tying_type must be specified when using weight_tying.")
        self.weight_tying_type = weight_tying_type
        self.config_latent_encoder_sampler = config_latent_encoder_sampler
        self.config_inference_sampler = config_inference_sampler
        self.latent_variable_average_states = latent_variable_average_states
        self.annealing = annealing


class SockeyeModel:
    """
    SockeyeModel shares components needed for both training and inference.
    The main components of a Sockeye model are
    1) Source embedding
    2) Target embedding
    3) Encoder
    4) Decoder
    5) Output Layer

    ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
    time.

    :param config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = copy.deepcopy(config)
        self.config.freeze()
        logger.info("%s", self.config)
        self.embedding_source = None  # type: Optional[encoder.Embedding]
        self.encoder = None  # type: Optional[encoder.Encoder]
        self.embedding_target = None  # type: Optional[encoder.Embedding]
        self.decoder = None  # type: Optional[decoder.Decoder]
        self.output_layer = None  # type: Optional[layers.OutputLayer]
        self._is_built = False
        self.params = None  # type: Optional[Dict]

    def save_config(self, folder: str):
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = ModelConfig.load(fname)
        logger.info('ModelConfig loaded from "%s"', fname)
        return config  # type: ignore

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.

        :param fname: Path to save parameters to.
        """
        assert self._is_built
        utils.save_params(self.params.copy(), fname)
        logging.info('Saved params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        assert self._is_built
        utils.check_condition(os.path.exists(fname), "No model parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % fname)
        self.params, _ = utils.load_params(fname)
        logger.info('Loaded params from "%s"', fname)

    @staticmethod
    def save_version(folder: str):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.VERSION_NAME)
        with open(fname, "w") as out:
            out.write(__version__)

    def _get_embed_weights(self) -> Tuple[
        mx.sym.Symbol, mx.sym.Symbol, mx.sym.Symbol]:
        """
        Returns embedding parameters for source and target.

        :return: Tuple of source and target parameter symbols.
        """
        assert isinstance(self.config.config_embed_source, encoder.EmbeddingConfig)
        assert isinstance(self.config.config_embed_target, encoder.EmbeddingConfig)
        utils.check_condition(self.config.weight_tying_type is None,
                              "Weight tying cannot be used together with output mixture model.")

        w_embed_source = mx.sym.Variable(C.SOURCE_EMBEDDING_PREFIX + "weight",
                                         shape=(self.config.config_embed_source.vocab_size,
                                                self.config.config_embed_source.num_embed))
        w_embed_target = mx.sym.Variable(C.TARGET_EMBEDDING_PREFIX + "weight",
                                         shape=(self.config.config_embed_target.vocab_size,
                                                self.config.config_embed_target.num_embed))
        output_layer_hidden_size = self.decoder.get_num_hidden()
        if self.config.config_decoder.latent_value_into_output:
            if self.config.config_decoder.stochastic_layer_size > 0:
                output_layer_hidden_size += self.config.config_decoder.stochastic_layer_size
        w_out_target = mx.sym.Variable("target_output_weight",
                                                shape=(self.config.vocab_target_size, output_layer_hidden_size))

        if self.config.weight_tying:
            if C.WEIGHT_TYING_SRC in self.config.weight_tying_type \
                    and C.WEIGHT_TYING_TRG in self.config.weight_tying_type:
                logger.info("Tying the source and target embeddings.")
                w_embed_source = w_embed_target = mx.sym.Variable(C.SHARED_EMBEDDING_PREFIX + "weight",
                                                                  shape=(self.config.config_embed_source.vocab_size,
                                                                         self.config.config_embed_source.num_embed))

            if C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type:
                logger.info("Tying the target embeddings and output layer parameters.")
                utils.check_condition(self.config.config_embed_target.num_embed == self.decoder.get_num_hidden(),
                                      "Weight tying requires target embedding size and decoder hidden size " +
                                      "to be equal: %d vs. %d" % (self.config.config_embed_target.num_embed,
                                                                  self.decoder.get_num_hidden()))
                w_out_target = w_embed_target

        return w_embed_source, w_embed_target, w_out_target

    def _build_model_components(self):
        """
        Instantiates model components.
        """
        # encoder & decoder first (to know the decoder depth)
        self.encoder = encoder.get_encoder(self.config.config_encoder)
        self.decoder = decoder.get_decoder(self.config.config_decoder)

        # source & target embeddings
        embed_weight_source, embed_weight_target, out_weight_target = self._get_embed_weights()
        assert isinstance(self.config.config_embed_source, encoder.EmbeddingConfig)
        assert isinstance(self.config.config_embed_target, encoder.EmbeddingConfig)
        self.embedding_source = encoder.Embedding(self.config.config_embed_source,
                                                  prefix=C.SOURCE_EMBEDDING_PREFIX,
                                                  embed_weight=embed_weight_source)
        self.embedding_target = encoder.Embedding(self.config.config_embed_target,
                                                  prefix=C.TARGET_EMBEDDING_PREFIX,
                                                  embed_weight=embed_weight_target)

        self.latent_encoder_sampler = None
        if self.config.config_latent_encoder_sampler:
            self.latent_encoder_sampler = rep.get_reparametrisation_sampler(self.config.config_latent_encoder_sampler)
        self.inference_latent_sampler = None
        if self.config.config_inference_sampler:
            # Block gradient flow into target embeddings
            self.inference_target_encoder = encoder.get_encoder(self.config.config_encoder,
                                                                prefix=C.TARGET_ENCODER_PREFIX)
            self.inference_latent_sampler = rep.get_reparametrisation_sampler(self.config.config_inference_sampler,
                                                                              prefix=C.INFERENCE_PREFIX)
        if self.config.config_decoder.variational_training:
            self.backward_target_encoder = encoder.get_encoder(
                self.config.config_encoder.copy(reverse_input=True, unidirectional=True),
                prefix=C.BACKWARD_TARGET_ENCODER_PREFIX)

        # output layer
        output_layer_hidden_size = self.decoder.get_num_hidden()
        if self.config.config_decoder.latent_value_into_output:
            output_layer_hidden_size += self.config.config_decoder.stochastic_layer_size
        self.output_layer = layers.OutputLayer(hidden_size=output_layer_hidden_size,
                                               vocab_size=self.config.vocab_target_size,
                                               weight=out_weight_target,
                                               weight_normalization=self.config.weight_normalization)

        self._is_built = True