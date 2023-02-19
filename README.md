# LEARNING-MULTI-RATE-VECTOR-QUANTIZATION-FOR-REMOTE-DEEP-INFERENCE
Multi-rate remote deep inference scheme, which trains a single encoder-decoder model that uses learned vector quantizers while supporting different quantization levels.

This is a PyTorch implementation of the 'Learning Multi-Rate Vector Quantization for Remote Deep Inference', by May Malka, Shai Ginzach, and Nir Shlezinger.

## Running

To run the code simply run `AdaptiveCodebook.py`. You can also add parameters in the command line. The default values are defined at the Globals&Hyperparameter section in AdaptiveCodebook.py.

## Models

The Adaptive VQ-VAE has the following fundamental model components:

1. An `Encoder` which apply the mapping `x -> x_e`
2. A `AdpativeVectorQuantizer` which quantize the encoder output `x_e -> z` by the 'euclidean distance' measure.
3. A `Decoder` class which maps `z -> y_hat`.

The architecture of the network is taken from the mobilenetv2 architecture: https://arxiv.org/abs/1801.04381.


