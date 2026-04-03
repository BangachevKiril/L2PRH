Training top-K sparse autoencoders on the embeddings of representations and processing them. 

- script top_sae.py and topk_sae_bash.sh perform the main task of training sparse autoencoders. The method is top-k sae with dead-neuron resampling, similar to this
  [Transformers Circuit Thread](https://transformer-circuits.pub/2023/monosemantic-features), for example. Detailed pseudo-code is given in the assocoiated paper.

- sparse_truncation.py and sparse_truncation_bash.sh filter the features by removing features that appear too rarely or too frequently. Currently features that appear in >10% of 
the representations or less that 0.001% are removed. The first are treated as polysemantic, the latter as noise.

- compute_sae_residuals.py and compute_sae_residuals_bash.sh compute the SAE residuals.

- compute_incoherence_statistics.py and compute_incoherence_statistics_bash.sh compute statistics of the dictionary A produced by the SAE. Concretely, if A\in R^{D \times d}
where D is the dictionary size and d the dense representation dimension, analyzed are the off-diagonal entries of A^tA. Their empirical distribution
is related to the (in)coherence of the dictionary A.

 - compute_sparse_feature_statistics.py and compute_sparse_feature_statistics_bash.sh compute statistics of the non-zero entries of the sparse features. This is done to empirically
   support the claim that non-zero features are of roughly similar magnitude.
