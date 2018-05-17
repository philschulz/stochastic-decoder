# stochastic-decoder
This repository allows for reproduction of the experiments reported in [Schulz et al (2018): A Stochastic Decoder for Neural Machine Translation](). The implementation is based on [Sockeye](https://github.com/awslabs/sockeye). Please check their documentation for the basic command line arguments.

Our implementation adds latent variables to the decoder. The basic way to enable these is to provide the following command line arguments:
```--latent-dim <dim> --use-variational-approximation --decoder-rnn-stochastic```
The first argument specifies the size (dimensionality) of the latent variables. The third argument is optional. If it is not used, a single sentence-level latent variable will be used. With that argument there will be one latent variable per target position.

## Code Status

At the moment this repository contains the code that we used for the experiments reported in our ACL paper. It is terribly outdated with respect to Sockeye. We will soon provide a version that has been cleaned and rebased on the most current version of Sockeye. We also hope to make the stochastic decoder a part of Sockeye soon.

## Running Experiments with the Workflow

### Installing Dependencies

We have tried to make experimenting with our code as convenient as possible but please feel free to reach out if you have any questions. First, run ```install_dependencies.sh <installer>```. This will create a Python virtual environment called *stochastic-decoder-env* and install our version of Sockeye as well as the necessary dependencies. It will also install some libraries for which root access is needed (those are required to install [sentencepiece](https://github.com/google/sentencepiece) and [multeval](https://github.com/jhclark/multeval) which we use as part of our workflow).

### Setting Parameters

Once the installation is finished, you can use our workflow to run experiments. Our workflow is implemented with [ducttape](https://github.com/jhclark/ducttape). There are different *plans* that one can run. You are free to define your own. We provide three plans:
* baseline: Train basline sockeye models with and without dropout.
* sent: Stochastic model with only one sentence-level latent variable.
* sdec: The stochastic decoder model as presented in the paper.

The training parameters can be changed in the [sockeye.tconf](workflow/sockeye.tconf). You may want to change *devices*. By default it is set to -2 which means that Sockeye will grab any two avaialable GPUs. You can select specific gpus by providing a white-space-seperated string of IDs (e.g. "2 5"). When you change the total number of GPUs, make sure to also change the ```batch_size```. We set it to 50 x number of GPUs.

### Running the Workflow

Make sure to use the virtualenv!
```source ~/stochastic-decoder-env/bin/activate```
Then run the workflow.
```ducttape workflow.tape -C sockeye.tconf -p <plan> -O <output_directory>```
The plan is one of *baseline, sent, sdec* or a custom plan. If you don't specify an output directory, the output will be stored in the current directory. When specifying an output directory use absolute paths as ducttape sometimes struggles with relative ones.

Once the workflow has finished you can use ```get_results.sh``` to list all evaluation scores produced by multeval.
