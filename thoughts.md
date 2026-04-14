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

Change SAA return calculation to a "solvable" POMDP problem. This means:
- Include all metrics used to calculate the reward in the observations
- Include the previous action in the observation to enhance path understanding of the agent

- Check the magnitude of the rewards. PPO tends to work best in the range of rewards being in a 1 range. Simply apply e.g. 100 factor if rewards are in 0.0001 range. 

- Use log prices in return calculation to try for positive explained variance.

- The simple weight observations of cash and assets is likely too simple and experiencing a drift. This is due to the fact that weights are dependent of the development of the other assets as well. 

- Switch cash weight to log(cash/total value) OR log(cash/starting funds)

- Route critical info: last action, log cash ratio, asset size past the mlp so it cannot dilute the information. Active asset size can be expressed as asset log(notional/starting funds). This should reduce relative drift introduced by other assets.

- Value loss should be in range 0.1 to 1 to comfortably learn