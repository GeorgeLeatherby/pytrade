VERY IMPORTANT: Training stopped due to reward and mean reward not improving. What about the excess return over SPY? Is this not considered? Also it seems that there is no last checkpoint is saved, when early stopping condition is identified. THIS needs to be fixed before further implementation can be conducted! Continue run \00281_config_10019_26_10_02 at it best excess SPY checkpoint. 

OPEN: Check whether caching of data that is loaded from the web can be enabled. It seems that there are still some web calls made when initializing


OPEN: Describe the exact type of transformer used for the PAA and explain why this was choosen. PAA is an encoder-only transformer structure. PAA does not generate a sequence of trades each step. It generates a rebalancing of the portfolio each step. Desired output is therefore a joint allocation vector. Actor uses asset tokens critic uses portfolio token. Architecture is NOT perfectly permutation-invariant, because learned asset identity mbeddings are included. Therefore the PAA is an encoder-only, identity-aware, cross-sectional Transformer actor-critic. 

OPEN: Check if the described feature description in the thesis is actually accurate. The PAA may receive more information from the SAA then is currently described. This is important for later argumentation in the discussion section regarding the effect ofthe SAA.

