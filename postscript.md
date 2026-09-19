# Postscript

In the original TensorFlow implementation, Yoon, the author, did not apply any transformation to the stock prices before using them as input for TimeGAN. I have preserved this setup in the [example notebook](example.ipynb) for fidelity to the original implementation, although this is divergent from common practices in finance. Raw equity prices can often vary widely in scale and returns – manifesting a more stable behavior across time – are typically preferred.

I have also carried out a stock-generating experiment with an additional log-return transformation step during data preprocessing. However, this does not lead to any improvement at all; instead, the generated sequences are absolutely terrible and they all look alike in candlestick charts. Output visualizations, namely, PCA and T-SNE, suggests otherwise though. The enhanced difficulty in learning from the noise-like return series is a likely culprit for the performance degradation.

Finding the right sequence length to cut your data also poses a challenge. One might reasonably expect asking TimeGAN to generate sequences of 24 observations to be a sensible setup. Yet all I observed among the resulting candlestick charts were kangaroo tails: long, monotone, mostly continuous-ish curvatures that taper either upward or downward. The only novelty I get among the trials is that windows of length 3 produces barely palatable results, as shown, again, in [example](example.ipynb).

Along with GANs' well-documented inherent pitfalls, these observations have led me to conclude that, under the experimental setup explored here, TimeGAN is not a suitable choice for generating realistic equity-price sequences. It has failed to produce authentic and convincing sequences – in which my eyes serve as the post-hoc discriminator.

That said, I'm more than happy to be proved wrong.

Boris
19 Sep 2026
