## Thoughts

The agent is really only learning something, if the return difference after initialisation constantly is positive. This is the actual alpha the agent can generate. Niels suggested to also challenge this with being fully invested. To kind of check against simply buying and holding the selected asset fully as well.

If still nothing learned in PAA:
Verify that loaded inference SAA LSTM produce actionable output in setup. Load them in same manner as for PAA and assign XEUR to each asset and then follow the outputs given the training parameters!
# TODO: Actually check this next also for verification purposes!


Tweak reward function further. Make it easier, make it more complex, see what works. Write each component of the reward into a tb_metric, so you can verify magnitudes, changes etc.

# Make sure to still comply with (PO)MDP premises!
Add as many as possible metrics to portfolio obs which are used to calculate the reward.

Enhance tb metrics. Show min/max/mean/median of action logits.