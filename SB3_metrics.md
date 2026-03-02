## SB3 Metrics ##

- time/fps: frames per second; speed of environment steps. Higher is faster training throughput.
- time/iterations: number of rollout iterations completed. Each iteration collects n_steps per env then updates.
- time/time_elapsed: seconds since training started.
- time/total_timesteps: environment steps consumed so far across all iterations.

- train/approx_kl: approximate KL divergence between old and new policies. Small values indicate conservative updates; too high can imply policy instability. 
- train/clip_fraction: fraction of policy gradient terms that were clipped by PPO’s ratio clamp. Near 0 means updates are within the trust region; large values mean frequent clipping.
- train/clip_range: current PPO clip hyperparameter. It’s scheduled via config defined clip schedule 
- train/entropy_loss: negative policy entropy; more negative implies higher entropy (more exploration). As policy converges, magnitude should decrease.
- train/explained_variance: EV of the value function over returns; near 1 is good fit, near 0 or negative indicates poor value learning (common early in training).
- train/learning_rate: current LR from the schedule. Starts at learning_rate_start and will decay toward learning_rate_end.
- train/loss: overall training loss (sum of policy, value, and entropy components per SB3 implementation). Large early values are normal; trend matters.
- train/n_updates: total gradient update batches applied so far.
- train/policy_gradient_loss: PPO policy loss; negative indicates improvement (since PPO minimizes loss that includes negative advantages).
- train/std: action distribution standard deviation (for Gaussian policy). Higher std → more exploration; it often shrinks as learning progresses.
- train/value_loss: critic (value function) MSE. Large early, should decrease as value fits returns.