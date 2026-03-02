
import numpy as np
from typing import Dict, Any, List
from obsolete.trading_env import TradingEnv, TradeInstruction, EXECUTION_TRANCHE

BENCHMARK_WEIGHTS = {
    "SPY": 0.45,
    "Gold": 0.20,
    "Crude": 0.05,
    "EWJ": 0.10,
    "EWG": 0.10,
    "EWQ": 0.05,
    "EWT": 0.05
}

def _build_env(config: Dict[str, Any], cache, mode: str):
    return TradingEnv(config, cache, mode=mode)

def _make_buy_instructions(cache, prices: np.ndarray, portfolio_value: float) -> List[Dict[str, Any]]:
    instr = []
    for sym, w in BENCHMARK_WEIGHTS.items():
        if sym not in cache.asset_to_index:
            print(f"Warning: Benchmark symbol {sym} not in market data cache; skipping BUY instruction.")
            continue
        idx = cache.asset_to_index[sym]
        price = float(prices[idx])
        if price <= 0 or not np.isfinite(price):
            print(f"Warning: Invalid price for {sym} ({price}); skipping BUY instruction.")
            continue
        target_notional = portfolio_value * w
        shares = int(target_notional // price)
        if shares > 0:
            instr.append({
                "symbol": sym,
                "action": "BUY",
                "quantity": float(shares),
                "order_type": "MARKET"
            })
            print(f"  Created entry: BUY {sym} qty={shares} for target notional {target_notional:.2f} at price {price:.2f}")
    return instr

def _make_sell_instructions(cache, positions: np.ndarray) -> List[Dict[str, Any]]:
    instr = []
    for sym in BENCHMARK_WEIGHTS.keys():
        if sym not in cache.asset_to_index:
            print(f"Warning: Benchmark symbol {sym} not in market data cache; skipping SELL instruction.")
            continue
        idx = cache.asset_to_index[sym]
        held = float(positions[idx])
        if held > 0:
            instr.append({
                "symbol": sym,
                "action": "SELL",
                "quantity": held,
                "order_type": "MARKET"
            })
            print(f"  Created exit: SELL {sym} qty={held} for held position.")
    return instr

def _run_episode(env: TradingEnv, episode_id: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=episode_id)
    initial_val = info["initial_portfolio_value"]
    initial_cash = info["initial_cash"]
    start_date = info["initial_date"]
    block_id = info["block_id"]
    print(f"\n=== Episode {episode_id} START | Block {block_id} | Start Date {start_date} ===")
    print(f"Initial Portfolio Value: {initial_val:.2f} | Cash: {initial_cash:.2f}")

    all_trade_logs = []
    while True:
        if env.current_step == 0:
            # Buy benchmark allocations
            prices = env.portfolio_state.prices.copy()
            buy_instr = _make_buy_instructions(env.market_data_cache, prices, env.initial_portfolio_value)
            print(f"[Episode {episode_id}] BUY step (day {env.current_step}) instructions:")
            for x in buy_instr:
                print(f"  BUY {x['symbol']} qty={x['quantity']}")
            action = buy_instr
        elif env.current_step == env.episode_length_days - 1:
            # Sell everything (benchmark components only)
            sell_instr = _make_sell_instructions(env.market_data_cache, env.portfolio_state.positions.copy())
            print(f"[Episode {episode_id}] SELL step (day {env.current_step}) instructions:")
            for x in sell_instr:
                print(f"  SELL {x['symbol']} qty={x['quantity']}")
            action = sell_instr
        else:
            action = []  # Hold

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        # Capture trade results only when trades attempted
        if action:
            tr = step_info.get("trade_results", [])
            print(f"[Episode {episode_id}] TradeResults (step {step_info['step']}):")
            for r in tr:
                print(f"  {r['action']} {r['symbol']}: req={r['requested_qty']}, exec={r['executed_qty']}, "
                      f"px={r['execution_price']}, notion={r['notional']}, cost={r['transaction_cost']}, reason={r['reason']}")
            all_trade_logs.extend(tr)

        if terminated or truncated:
            final_val = step_info["portfolio_value"]
            port_ret = (final_val / initial_val) - 1.0
            print(f"=== Episode {episode_id} END | Final Value {final_val:.2f} | Return {port_ret:.4%} "
                  f"| Steps {step_info['step']} ===")
            break

    return {
        "episode": episode_id,
        "initial_value": initial_val,
        "final_value": final_val,
        "return": port_ret,
        "num_trades": sum(1 for r in all_trade_logs if r.get("success")),
        "trade_logs": all_trade_logs
    }

def run(cache, config) -> Dict[str, Any]:
    train_episodes = int(config.get("training_episodes", 10))
    validation_episodes = int(config.get("validation_episodes", 1))

    # Training
    env_train = _build_env(config, cache, mode='train')
    train_results = []
    for ep in range(train_episodes):
        res = _run_episode(env_train, ep)
        train_results.append(res)

    # Validation
    env_val = _build_env(config, cache, mode='validation')
    val_results = []
    for vep in range(validation_episodes):
        res = _run_episode(env_val, train_episodes + vep)
        val_results.append(res)

    print("\n=== Summary ===")
    for r in train_results:
        print(f"Train Episode {r['episode']}: Return {r['return']:.4%}, Trades {r['num_trades']}, Final {r['final_value']:.2f}")
    for r in val_results:
        print(f"Validation Episode {r['episode']}: Return {r['return']:.4%}, Trades {r['num_trades']}, Final {r['final_value']:.2f}")

    return {
        "train_episode_count": train_episodes,
        "validation_episode_count": validation_episodes,
        "train_results": train_results,
        "validation_results": val_results
    }