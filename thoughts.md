## Thoughts

The agent is really only learning something, if the return difference after initialisation constantly is positive. This is the actual alpha the agent can generate. Niels suggested to also challenge this with being fully invested. To kind of check against simply buying and holding the selected asset fully as well.

If still nothing learned in PAA:
Verify that loaded inference SAA LSTM produce actionable output in setup. Load them in same manner as for PAA and assign XEUR to each asset and then follow the outputs given the training parameters!
# TODO: Actually check this next also for verification purposes!


Tweak reward function further. Make it easier, make it more complex, see what works. Write each component of the reward into a tb_metric, so you can verify magnitudes, changes etc.

# Make sure to still comply with (PO)MDP premises!
Add as many as possible metrics to portfolio obs which are used to calculate the reward.

Enhance tb metrics. Show min/max/mean/median of action logits.


# Use proper purging (purge_length) with walkforward method & (normalization_length)
    Step1: elimante too large normalization windows from data calculation. Define a max length for all calculated features and use it everywhere. (E.g. purge_length=60)
    Step2: Only allow feature lookbacks up to purge_length!
    Step3: Make sure that normalization windows do not change because of warmup phases. Data normalized with stats below (normalization_length)

    Fragen für Normalisierung: Wenn jedes asset einzelnd normalisiert wird, geht information zwischen den assets verloren. Wie groß der jeweilige Preisschritt bei jedem asset ist müssten die agents durch den reward herausfinden. 

## Retrain the SAA LSTM on refit data!
SAA MLP  size is hardcoded not taken from config. I reduced it from 3 layers 128 to 3 layers 64, whilst also making lstm layer 64 from 128. A test would be to enhance the MLP layer again (not recommended by LLM)

Another approach would be to increase lstm layer size to 2! This might capture temporal patterns better. Literature to check according to LLM:
- Time Series Forecasting (Lai et al., 2018)
- Multi-Scale Temporal Processing (Chung et al., 2016)
- Deep RNNs for Long-Term Dependencies (Pascanu et al., 2013)
- Hierarchical Temporal Representation (Graves et al., 2013)
