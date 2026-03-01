Only implement what is requested.
Always reason and seek context of other code and conversation before implementing.
If there is an unclarity, instead of providing code, ask a question to clarify.
State all your assumptions.
Verify all handshakes. Verify all shapes of returned objects with the expected shapes.

Overall goal: Use PPO. Use Transformer. Apply Cross-asset/portfolio attention. Do not implement a custom policy for PPO. Use signal of stateful SAA in inference mode (frozen) as additional signal inside the asset tokens. Asset tokens contain: raw market features per asset, SAA signal, portfolio weight. There is a single SAA model which will be applied n-times (because there are n assets). SAA was trained using RecurrentPPO. It needs states and end of episode signals.

When working with environment incorporation, only introduce a wrapper if really necessary! If possible use or add to existing wrappers! Make sure to fully understand the relevant environment sections first, before starting to code. If there are unclarities, ask first.

Apply computationally efficient code. Stay in "torch-land" whenever possible. Avoid loops, when vectorized computation is possible.

Use in config file defined paramteres, always! If a new variable really needs to be introduced this needs to be clearly stated and explained in the answer.

Prefer simple implementations over complicated ones. 
Write the intended/needed shape in commentary next to objects when initiating or returning.
Stick to already used nomenclature.

Prefer sb3 and PyTorch onboard methods and functions.