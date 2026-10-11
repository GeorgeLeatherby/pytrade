VERY IMPORTANT: Training stopped due to reward and mean reward not improving. What about the excess return over SPY? Is this not considered? Also it seems that there is no last checkpoint is saved, when early stopping condition is identified. THIS needs to be fixed before further implementation can be conducted! Continue run \00281_config_10019_26_10_02 at it best excess SPY checkpoint. 

OPEN: Check whether caching of data that is loaded from the web can be enabled. It seems that there are still some web calls made when initializing


RESOLVED (2026-10-10): The current allocator is an encoder-only, identity-aware, cross-sectional Transformer actor-critic. The shared encoder processes all asset and portfolio tokens; a private actor encoder processes asset tokens and mean-pools them, while a private critic encoder processes the already contextualized portfolio token. It emits one joint rebalancing vector, not a future trade sequence. Learned identities, ordered portfolio fields, and fixed allocation-output positions prevent a general permutation-invariance claim. Chapter 3 and its diagram now explain this architecture.

RESOLVED (2026-10-10): The tokenizer and environment confirm 32 full-feature asset inputs (26 market + action + shadow fraction + live weight + recorded target + shadow Sortino + shadow drawdown) and 65 portfolio inputs (6 calendar + 12 live weights + 14 scalars + 33 per-asset targets/risk summaries). Shadow Sortino and drawdown enter both token types. No recurrent hidden state or explicit policy-confidence estimate is transferred. The zero-signal ablation retains risk-summary fields for its untraded cash accounts, so the results test the combined policy/shadow-account interface rather than an action-only contribution. Chapters 3-5 now reflect this.

