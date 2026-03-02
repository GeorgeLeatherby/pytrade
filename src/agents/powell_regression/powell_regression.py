import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
from scipy import optimize

from obsolete.trading_env import TradingEnv, TradeInstruction, EXECUTION_TRANCHE

class EarlyStop(Exception):
    pass

DEBUG = True  # Set to True to enable debug logging (bunch of prints)
DEBUG_EPISODE_LEVEL = False  # Set to True to enable episode-level debug prints

# -------- Rule variables (theta) here only default values are displayed --------
# theta order:
# 0: th_r5_up        [−0.05, 0.10]     (5-day return must exceed)
# 1: th_mom10_up     [−0.05, 0.10]     (10-day momentum must exceed)
# 2: th_bw_min       [0.01, 0.30]      (min Bollinger bandwidth)
# 3: th_r1_down      [−0.05, 0.05]     (1-day return below triggers SELL)
# 4: sl_pct          [0.03, 0.15]      (trailing stop from last buy)
# 5: tp_pct          [0.06, 0.30]      (take profit from last buy)
# 6: buy_notional_usd[1000, 100000]    (tranche notional per BUY)

# Initialize THETA_BOUNDS with default ranges. This will be overwritten from config!
THETA_BOUNDS_multistart = np.array([
    [-0.05, 0.10],   # th_r5_up
    [-0.05, 0.10],   # th_mom10_up
    [0.01, 0.30],    # th_bw_min
    [-0.05, 0.05],   # th_r1_down
    [0.03, 0.15],    # sl_pct
    [0.06, 0.30],    # tp_pct
    [1000, 100000]   # buy_notional_usd
], dtype=np.float32)

THETA_BOUNDS_refinement = np.array([
    [-0.02, 0.05],   # th_r5_up
    [-0.02, 0.05],   # th_mom10_up
    [0.01, 0.20],    # th_bw_min
    [-0.03, 0.03],   # th_r1_down
    [0.05, 0.12],    # sl_pct
    [0.10, 0.30],    # tp_pct
    [5000, 50000]    # buy_notional_usd
], dtype=np.float32)

def _save_ordered_capped_json(filepath: str, entries: List[Dict[str, Any]], key_name: str, max_entries: int = 200):
    # Merge with existing entries (append-only store across sessions)
    existing: List[Dict[str, Any]] = []
    p = Path(filepath)
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    merged = (existing + entries)

    # Sort and cap by the provided key (assumes key exists in entries appended here)
    try:
        entries_sorted = sorted(merged, key=lambda x: x[key_name], reverse=True)[:max_entries]
    except KeyError:
        # Fallback: tolerate older keys; try common variants
        def sort_key(x):
            for k in (key_name, "average_return", "validation_average_return"):
                if k in x and isinstance(x[k], (int, float)):
                    return x[k]
            return float("-inf")
        entries_sorted = sorted(merged, key=sort_key, reverse=True)[:max_entries]

    p.write_text(json.dumps(entries_sorted, indent=2))

def _append_multistart_entry(entries: List[Dict[str, Any]], avg_return: float, final_theta: np.ndarray, initial_theta: np.ndarray):
    entries.append({
        "average_return": float(avg_return),
        "final_theta": [float(x) for x in final_theta.tolist()],
        "initial_theta": [float(x) for x in initial_theta.tolist()],
    })

def _append_refined_entry(entries: List[Dict[str, Any]], val_avg_return: float, avg_return: float, theta: np.ndarray):
    entries.append({
        "validation_average_return": float(val_avg_return),
        "average_return": float(avg_return),
        "theta": [float(x) for x in theta.tolist()],
    })

def _apply_bounds_from_config(config: Dict[str, Any]) -> None:
    """
    Update global THETA_BOUNDS for multistart from config['theta_bounds_multistart'] if present.
    Must be called before generating initial guesses or clipping.
    """
    tb_multi_cfg = config.get("theta_bounds_multistart", None)
    if not tb_multi_cfg:
        return
    keys = ["th_r5_up","th_mom10_up","th_bw_min","th_r1_down","sl_pct","tp_pct","buy_notional_usd"]
    for i, key in enumerate(keys):
        if key in tb_multi_cfg:
            lo, hi = tb_multi_cfg[key]
            THETA_BOUNDS_multistart[i,0] = float(lo)
            THETA_BOUNDS_multistart[i,1] = float(hi)

    tb_refinement_cfg = config.get("theta_bounds_refinement", None)
    if not tb_refinement_cfg:
        return
    for i, key in enumerate(keys):
        if key in tb_refinement_cfg:
            lo, hi = tb_refinement_cfg[key]
            THETA_BOUNDS_refinement[i,0] = float(lo)
            THETA_BOUNDS_refinement[i,1] = float(hi)

def _within_bounds(theta: np.ndarray) -> np.ndarray:
    # Clip theta values to be within THETA_BOUNDS. 
    return np.clip(theta, THETA_BOUNDS_multistart[:,0], THETA_BOUNDS_multistart[:,1])

def _build_env(config: Dict[str, Any], cache, mode: str) -> TradingEnv:
    # Force tranche mode and current_close to ensure execution on same bar price
    cfg = dict(config)
    cfg_env = dict(cfg["environment"])
    cfg_env["execution_mode"] = "tranche"
    cfg_env["quantity_type"] = "shares"
    cfg_env["price_source"] = "current_close"
    cfg["environment"] = cfg_env
    return TradingEnv(cfg, cache, mode=mode)

def _extract_indices(cache) -> Dict[str, int]:
    need = ["return_5d", "momentum_10", "bb_width", "return_1d"]
    missing = [f for f in need if f not in cache.feature_to_index]
    if missing:
        raise RuntimeError(f"Missing required features in cache: {missing}")
    return {f: cache.feature_to_index[f] for f in need}

def _episode_metric(final_value: float, initial_value: float) -> float:
    # Use simple episode return as objective metric
    if initial_value <= 0:
        print("Warning: initial portfolio value <= 0 in episode metric calculation.")
        return 0.0
    return (final_value / initial_value) - 1.0

def _make_buy_sell_actions(
    cache,
    prices: np.ndarray,
    positions: np.ndarray,
    feature_slice: np.ndarray,
    idx_map: Dict[str, int],
    theta: np.ndarray,
    entry_price_last_buy: np.ndarray
) -> List[Dict[str, Any]]:
    th_r5_up, th_mom10_up, th_bw_min, th_r1_down, sl_pct, tp_pct, buy_notional_usd = theta
    actions: List[Dict[str, Any]] = []

    for a_idx, sym in enumerate(cache.asset_names):
        px = float(prices[a_idx])
        if px <= 0 or not np.isfinite(px):
            continue

        r5 = float(feature_slice[a_idx, idx_map["return_5d"]])
        mom10 = float(feature_slice[a_idx, idx_map["momentum_10"]])
        bw = float(feature_slice[a_idx, idx_map["bb_width"]])
        r1 = float(feature_slice[a_idx, idx_map["return_1d"]])

        held = float(positions[a_idx])

        if held <= 0:
            # BUY rule: momentum + bandwidth filter
            if (r5 > th_r5_up) and (mom10 > th_mom10_up) and (bw >= th_bw_min):
                # Enough cash check is done in env; we just submit tranche
                actions.append({
                    "symbol": sym,
                    "action": "BUY",
                    "quantity": None,
                    "notional": float(buy_notional_usd),
                    "order_type": "MARKET"
                })
        else:
            # SELL rule: 1d return breach OR trailing stop OR take profit
            if entry_price_last_buy[a_idx] > 0:
                perf = (px / float(entry_price_last_buy[a_idx])) - 1.0
            else:
                perf = 0.0
            sell_trigger = (r1 < th_r1_down) or (perf <= -sl_pct) or (perf >= tp_pct)
            if sell_trigger:
                actions.append({
                    "symbol": sym,
                    "action": "SELL",
                    "quantity": held,  # close position fully in shares
                    "notional": None,
                    "order_type": "MARKET"
                })

    return actions

def eval_one_episode(
    trading_env, # TradingEnv
    cache, # MarketDataCache
    base_config: Dict[str, Any], # base config dict
    theta: np.ndarray, # rule parameters
    episode_id: int, # episode identifier
    mode: str = "train", # mode of operation: "train", "validation", or "test"
    verbose: bool = False # verbosity flag
) -> float:
    
    # Generate random episode seed
    episode_seed = np.random.randint(0, 1_000)  

    # --- Reset TradingEnv for this episode ---
    obs, info = trading_env.reset(seed=episode_seed)

    # Get feature indices
    idx_map = _extract_indices(cache)
    # Track entry price for trailing stops
    entry_price_last_buy = np.zeros(cache.num_assets, dtype=np.float32)

    initial_val = float(info["initial_portfolio_value"])
    if DEBUG_EPISODE_LEVEL == True:
        if verbose:
            print(f"[Episode {episode_id}] start | date={info['initial_date']} | cash={info['initial_cash']:.2f}")

    # --- Episode loop ---
    done = False
    buy_trades_proposed = 0
    sell_trades_proposed = 0
    buy_trades_executed = 0
    sell_trades_executed = 0

    while not done:
        abs_step = trading_env.current_absolute_step
        # Current features slice for all assets at absolute step
        feats = cache.get_features_at_step(abs_step)  # shape [A, F]
        prices = trading_env.portfolio_state.prices.copy()
        positions = trading_env.portfolio_state.positions.copy()

        actions = _make_buy_sell_actions(
            cache=cache,
            prices=prices,
            positions=positions,
            feature_slice=feats,
            idx_map=idx_map,
            theta=theta,
            entry_price_last_buy=entry_price_last_buy
        )

        # Count proposed trades
        for act in actions:
            if act["action"] == "BUY":
                buy_trades_proposed += 1
            elif act["action"] == "SELL":
                sell_trades_proposed += 1

        if DEBUG_EPISODE_LEVEL == True:
            # Log actions and portfolio state
            print(f" Step: {trading_env.current_step} | Abs Step: {abs_step} | Actions: {actions} | Cash: {trading_env.portfolio_state.cash:.2f} | Total Value: {trading_env.portfolio_state.get_total_value():.2f}")
            print(f" Positions in portfolio: {trading_env.portfolio_state.positions}")

        # --- Advance a day inside episode and propose trades ---
        next_obs, reward, terminated, truncated, step_info = trading_env.step(actions)

        # Update entry prices after successful BUYs
        if actions:
            trade_results = step_info.get("trade_results", [])
            for tr in trade_results:
                if tr.get("success"):
                    sym = tr["symbol"]
                    a_idx = cache.asset_to_index.get(sym, None)
                    if a_idx is None:
                        continue
                    if tr["action"] == "BUY" and tr["executed_qty"] > 0:
                        # Set/refresh last entry price to execution price
                        entry_price_last_buy[a_idx] = float(tr["execution_price"])
                        buy_trades_executed += 1
                    elif tr["action"] == "SELL" and tr["executed_qty"] > 0:
                        sell_trades_executed += 1

        # Termination of while loop
        done = bool(terminated or truncated)

    final_val = float(trading_env.portfolio_state.get_total_value())
    metric = _episode_metric(final_val, initial_val)
    if DEBUG_EPISODE_LEVEL == True:
        if verbose:
            print(f"[Episode {episode_id}] end | final={final_val:.2f} | return={metric:.4%}")
    return metric, buy_trades_proposed, sell_trades_proposed, buy_trades_executed, sell_trades_executed

def averaged_objective(
    trading_env, # TradingEnv
    K_batch: int, # number of episodes per evaluation batch
    cache, # MarketDataCache
    base_config: Dict[str, Any], # base config dict
    theta: np.ndarray, # rule parameters
    mode: str = "train",
    print_prefix: str = "",
    early_stop_state: Dict[str, Any] = None # optional state dict for early stopping per run
) -> float:
    
    metrics = []
    total_buy_trades_proposed = 0
    total_sell_trades_proposed = 0
    total_buy_trades_executed = 0
    total_sell_trades_executed = 0

    for eps in range(K_batch):
        objective_single_episode, buy_trades_proposed, sell_trades_proposed, buy_trades_executed, sell_trades_executed = eval_one_episode(trading_env, cache, base_config, theta, episode_id=eps, mode=mode, 
                                                    verbose=False
        )
        metrics.append(objective_single_episode)

        # Keeping track of total trades across episodes
        total_buy_trades_proposed += buy_trades_proposed
        total_sell_trades_proposed += sell_trades_proposed
        total_buy_trades_executed += buy_trades_executed
        total_sell_trades_executed += sell_trades_executed

    avg_metric = float(np.mean(metrics)) if metrics else 0.0

    if not metrics: print("Warning: No metrics computed in averaged_objective.")
    if DEBUG == True:
        print(f"{print_prefix}f(theta) -> avg_return={avg_metric:.4%} over {len(metrics)} episodes | theta={theta}")
        print(f"    Buy trades proposed: {total_buy_trades_proposed}, executed: {total_buy_trades_executed}")
        print(f"    Sell trades proposed: {total_sell_trades_proposed}, executed: {total_sell_trades_executed}")

    # NEW: update per-run early-stop state
    if early_stop_state is not None:
        total_proposed = total_buy_trades_proposed + total_sell_trades_proposed
        if total_proposed == 0:
            early_stop_state['consec_no_trade'] = early_stop_state.get('consec_no_trade', 0) + 1
        else:
            early_stop_state['consec_no_trade'] = 0
        early_stop_state['max_avg_return'] = max(early_stop_state.get('max_avg_return', -np.inf), avg_metric)

    return -avg_metric # minimize negative average return

def _random_initial_theta(n_starts: int) -> List[np.ndarray]:
    inits = []
    for _ in range(n_starts):
        r = np.random.rand(THETA_BOUNDS_multistart.shape[0])
        theta0 = THETA_BOUNDS_multistart[:,0] + r * (THETA_BOUNDS_multistart[:,1] - THETA_BOUNDS_multistart[:,0])
        inits.append(theta0.astype(np.float32))
    return inits

def _powell_minimize(
    f_callable,
    bounds,
    theta0: np.ndarray,
    maxiter: int,
    xtol: float,
    ftol: float,
):
    # Build a small initial direction set to tone down early steps.
    # Each column is a direction; start with identity scaled by small step sizes.
    n = len(theta0)
    small_steps = np.array([  # per-theta gentle step sizes
        0.01,  # th_r5_up
        0.01,  # th_mom10_up
        0.005,  # th_bw_min
        0.01,  # th_r1_down
        0.02,  # sl_pct
        0.02,  # tp_pct
        1000.0,  # buy_notional_usd (smaller initial move in $)
    ], dtype=np.float64)
    direc = np.eye(n, dtype=np.float64)
    direc = (direc.T * small_steps).T  # scale identity columns

    res = optimize.minimize(
        fun=f_callable,
        x0=np.asarray(theta0, dtype=np.float64),
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "xtol": xtol, "ftol": ftol, "disp": False, "direc": direc}
    )
    return res

def run(cache, config) -> Dict[str, Any]:

    # Settings from config ----------------------------------------------------
    # Ensure bounds reflect config before creating initial guesses or clipping
    _apply_bounds_from_config(config)
    # Get settings from config
    K_train_batch = int(config.get("optimization", {}).get("episodes_per_eval", 5)) # episodes per training eval
    K_val_batch = int(config.get("optimization", {}).get("episodes_per_validation", 8)) # episodes per validation eval
    n_starts = int(config.get("optimization", {}).get("multi_start", 6)) # number of random starts
    maxiter = int(config.get("optimization", {}).get("maxiter", 200)) # max iterations for optimizer
    xtol = float(config.get("optimization", {}).get("xtol", 1e-3)) # tolerance for x convergence
    ftol = float(config.get("optimization", {}).get("ftol", 1e-3)) # tolerance for function convergence

    # get early stopping measure from config
    early_stop_consec_no_trade = int(config.get("multistart_early_stopping", {}).get("consec_no_trade", 10))
    early_stop_min_avg_return = float(config.get("multistart_early_stopping", {}).get("min_avg_return", 0.02))

    # Config for refinement selection
    refine_select_top = int(config.get("optimization", {}).get("refine_select_top", 10))
    refine_min_avg = float(config.get("optimization", {}).get("refine_min_avg_return", 0.02))

    # Setup initial guesses -------------------------------------------------
    # Draw initial guess set based on config ranges (optional override)
    init_cfg = config.get("initial_guess_ranges", {})
    use_config_inits = bool(init_cfg)
    if use_config_inits:
        # Build initial guesses from config ranges
        initial_guesses = []
        for _ in range(n_starts):
            theta = []
            for i in range(THETA_BOUNDS_multistart.shape[0]):
                key = ["th_r5_up","th_mom10_up","th_bw_min","th_r1_down","sl_pct","tp_pct","buy_notional_usd"][i]
                lo, hi = init_cfg.get(key, [float(THETA_BOUNDS_multistart[i,0]), float(THETA_BOUNDS_multistart[i,1])])
                val = lo + np.random.rand() * (hi - lo)
                theta.append(val)
            initial_guesses.append(np.array(theta, dtype=np.float32))
        print(f"Used config-based initial guesses for Powell optimization.")
    else:
        initial_guesses = _random_initial_theta(n_starts)
        print(f"Warning: Used random initial guesses for Powell optimization.")

    # Init the Trading Environment ---------------------------------
    trading_env = _build_env(config, cache, mode="train")

    # Multi-start optimization
    best_theta = None
    best_obj = None

    # Build a bounds object for scipy.optimize based on THETA_BOUNDS
    multistart_bounds = optimize.Bounds(THETA_BOUNDS_multistart[:,0], THETA_BOUNDS_multistart[:,1])

    # Accumulators
    multistart_results: List[Dict[str, Any]] = []
    selected_for_refinement: List[Dict[str, Any]] = []

    # Loop through random start values of theta
    for start_idx, theta0 in enumerate(initial_guesses):
        print(f"\n=== Powell run {start_idx+1}/{n_starts} | K={K_train_batch} train episodes | theta0={theta0} ===")

        early_stop_state = {'consec_no_trade': 0, 'max_avg_return': -np.inf}

        # Objective closure with clipping and fixed episodes
        def f(theta_vec: np.ndarray):
            # ensure theta within bounds
            theta_vec = _within_bounds(np.asarray(theta_vec, dtype=np.float32))
            # Get averaged objective over fixed episodes
            averaged_objective_value = averaged_objective(
                trading_env, K_train_batch, cache, config, theta_vec, mode="train", print_prefix=f"[run {start_idx+1}] ",
                early_stop_state=early_stop_state)
            
            # NEW: early-stop check (10 consecutive no-trade batches AND no avg return >= X%)
            if early_stop_state['consec_no_trade'] >= early_stop_consec_no_trade and early_stop_state['max_avg_return'] < early_stop_min_avg_return:
                raise EarlyStop(f"Early stop: run {start_idx+1} had {early_stop_state['consec_no_trade']} consecutive no-trade evals and max avg return {early_stop_state['max_avg_return']:.4%} < {early_stop_min_avg_return:.4%}.")
            return averaged_objective_value

        # --- Run Powell optimization ---
        # Wrapped to catch early stop
        try:
            res = _powell_minimize(f, multistart_bounds, theta0, maxiter=maxiter, xtol=xtol, ftol=ftol)
            final_theta = res.x.astype(np.float32) # no more clipping to bounds here after optimization
            final_obj = float(res.fun)
            final_avg_return = -final_obj

            print(f"avg_return={final_avg_return:.4%} | message: {res.message}")
            print(f"theta={final_theta} | iterations used ={res.nit}")

            # Track best from all starts for init validation and refinement
            if best_obj is None or final_obj < best_obj:
                best_obj = final_obj
                best_theta = final_theta

            # Collect multi-start result
            _append_multistart_entry(multistart_results, avg_return=final_avg_return, final_theta=final_theta, initial_theta=theta0)

            # Collect candidates for refinement (meeting minimum avg)
            if final_avg_return >= refine_min_avg:
                selected_for_refinement.append({
                    "avg_return": final_avg_return,
                    "theta": final_theta,
                    "theta0": theta0
                })

        except EarlyStop as e:
            print(str(e))
            print(f"Skipping remainder of run {start_idx+1} due to early-stop condition.")
            # Do not update best_theta/best_obj here; continue to next start
            continue

    # Save multistart results to file
    _save_ordered_capped_json("best_multistart_runs.json", multistart_results, key_name="average_return", max_entries=200)

    # TODO: Do this for all of the best runs!
    # Run init validation using the best theta found
    if best_theta is not None:
        print("\n=== Initial validation ===")
        # Use theta on K_val_batch validation episodes and generate summary
        val_metrics = []
        for vep in range(K_val_batch):
            m, *_ = eval_one_episode(trading_env, cache, config, best_theta, vep, mode="validation", verbose=True)
            val_metrics.append(m)
        val_avg = float(np.mean(val_metrics))
        print(f"avg_return={val_avg:.4%} over {K_val_batch} episodes")
        print(f"theta: {best_theta}")

    else:
        print("="*20)
        print("{\nNo best theta found from optimization. Skipping validation and refinement.")
        return None


    # --- Optional refinement with larger K and opt params ---
    print("\n=== Refinement stage (larger K) ===")
    refine_K = int(config.get("optimization", {}).get("refine_K", 10))

    # Select top candidates for refinement
    selected_for_refinement = sorted(selected_for_refinement, key=lambda x: x["avg_return"], reverse=True)[:refine_select_top]

    # Build refinement bounds object for scipy.optimize based on THETA_BOUNDS_refinement
    refinement_bounds = optimize.Bounds(THETA_BOUNDS_refinement[:,0], THETA_BOUNDS_refinement[:,1])

    # Accumulator for refined entries
    best_refined_entries: List[Dict[str, Any]] = []

    # Loop through selected candidates
    for idx, cand in enumerate(selected_for_refinement, start=1):
        print(f"\n--- Refinement {idx}/{len(selected_for_refinement)} | start avg={cand['avg_return']:.4%} ---")

        # Objective closure for refinement. Contains no clipping, uses larger K
        def f_refinement(theta_vec: np.ndarray):
            # Get averaged objective over fixed episodes
            averaged_objective_value = averaged_objective(trading_env, refine_K, cache, config, theta_vec, mode="train", print_prefix=f"[refinement] ")
            return averaged_objective_value
    
        # --- Run Powell optimization in refinement mode using candidates theta ---
        res = _powell_minimize(f_refinement, refinement_bounds, cand["theta"], maxiter=(maxiter*2), xtol=(xtol), ftol=(ftol))
        refined_theta = res.x.astype(np.float32) # no more clipping to bounds here after optimization
        final_obj = float(res.fun)
        refined_avg_return = -final_obj

        print(f"--- Refinement result: avg_return={refined_avg_return:.4%} | message: {res.message} | theta={refined_theta} | iters={res.nit} ---")

        # Validation of refined theta
        val_metrics = []
        for vep in range(K_val_batch):
            m, *_ = eval_one_episode(trading_env, cache, config, refined_theta, vep, mode="validation", verbose=False)
            val_metrics.append(m)
        val_avg_return = float(np.mean(val_metrics))

        # Track best from all starts for init validation and refinement
        if best_obj is None or final_obj < best_obj:
            best_obj = final_obj
            best_theta = refined_theta

        # Append refined entry
        _append_refined_entry(best_refined_entries, val_avg_return=val_avg_return, avg_return=refined_avg_return, theta=refined_theta)

    # Persist best refined runs capped and ordered
    _save_ordered_capped_json("best_refined_runs.json", best_refined_entries, key_name="validation_average_return", max_entries=200)

    print("\n=== Optimization complete ===")
    print(f"Best avg return: {-best_obj:.4%}")
    print(f"Best theta: {best_theta}")

    # Final validation summary
    if best_theta is not None:
        print("\n=== Initial validation ===")
        # Use theta on K_val_batch validation episodes and generate summary
        val_metrics = []
        for vep in range(K_val_batch):
            m, *_ = eval_one_episode(trading_env, cache, config, best_theta, vep, mode="validation", verbose=True)
            val_metrics.append(m)
        final_val_avg = float(np.mean(val_metrics))
        print(f"avg_return={final_val_avg:.4%} over {K_val_batch} episodes")
        print(f"theta: {best_theta}")

    else:
        print("="*20)
        print("{\nNo best theta found from optimization. Skipping validation and refinement.")
        return None


    return {
        "best_theta": best_theta.tolist() if best_theta is not None else None,
        "best_avg_return": float(-best_obj) if best_obj is not None else None,
        "final_validation_avg_return": final_val_avg,
        "theta_bounds": THETA_BOUNDS_multistart.tolist(),
        "optimizer": "Powell",
        "details": {
            "n_starts": n_starts,
            "episodes_per_eval": K_train_batch,
            "episodes_per_validation": K_val_batch
        }
    }