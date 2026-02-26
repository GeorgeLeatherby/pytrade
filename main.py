import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import json, importlib
import pandas as pd
from tkinter import Tk, Listbox, SINGLE, MULTIPLE, Button, Label, END, Toplevel, messagebox
from src.environment.trading_env import MarketDataCache

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "src", "agents")

def discover_agents():
    agents = []
    for entry in os.listdir(AGENTS_DIR):
        full = os.path.join(AGENTS_DIR, entry)
        if os.path.isdir(full):
            py_files = [f for f in os.listdir(full) if f.endswith(".py")]
            if py_files:
                agents.append(entry)
    return sorted(agents)

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def load_market_data(csv_path):
    return pd.read_csv(csv_path)

def run_agent(agent_folder, config_path):
    """
    This function dynamically loads and runs the specified agent with the given configuration based on the .py file.
    It loads market data, initializes the MarketDataCache, and calls the agent's run function
    """
    agent_dir = os.path.join(AGENTS_DIR, agent_folder)
    py_files = [f for f in os.listdir(agent_dir) if f.endswith(".py")]
    if not py_files:
        raise RuntimeError(f"No agent implementation file in {agent_dir}")
    module_name = py_files[0].rsplit(".", 1)[0]
    mod = importlib.import_module(f"src.agents.{agent_folder}.{module_name}")
    config = load_config(config_path)

    # Load market data from CSV into dataframe
    market_data_path = config.get("market_data_path") or os.path.join(os.path.dirname(__file__), "src", "data", "enriched_financial_data.csv")
    df = load_market_data(market_data_path)

    # Initialize MarketDataCache container from dataframe
    cache = MarketDataCache.from_dataframe(
        df, config,
        lookback_window=config["environment"]["lookback_window"],
        maybe_provide_sequence=config["environment"].get("maybe_provide_sequence", False)
    )

    # Run the agent's main function which needs to be called 'run'
    if hasattr(mod, "run"):
        result = mod.run(cache, config)
    else:
        raise RuntimeError(f"Agent module {agent_folder} missing a run(cache, config) function.")
    print(f"\n=== Agent: {agent_folder} | Config: {os.path.basename(config_path)} ===")
    print(result)

def select_agent_and_configs_gui():
    root = Tk(); root.title("Select Agent")
    agents = discover_agents()
    Label(root, text="Agents:").pack(padx=10, pady=5)
    lb = Listbox(root, selectmode=SINGLE, width=40, height=min(15, len(agents)))
    for a in agents: lb.insert(END, a)
    lb.pack(padx=10, pady=5)

    selected = {"agent": None, "configs": []}

    def on_select_agent():
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Selection", "Pick an agent.")
            return
        agent = agents[sel[0]]
        selected["agent"] = agent
        cfg_win = Toplevel(root); cfg_win.title(f"Configs for {agent}")
        agent_dir = os.path.join(AGENTS_DIR, agent)
        jsons = [f for f in os.listdir(agent_dir) if f.lower().endswith(".json")]
        Label(cfg_win, text="Configs: (multi-select)").pack(padx=10, pady=5)
        cfg_lb = Listbox(cfg_win, selectmode=MULTIPLE, width=60, height=min(20, max(1, len(jsons))))
        for jf in jsons: cfg_lb.insert(END, jf)
        cfg_lb.pack(padx=10, pady=5)
        def on_start():
            selc = cfg_lb.curselection()
            if not selc:
                messagebox.showwarning("Selection", "Pick at least one config.")
                return
            selected["configs"] = [os.path.join(agent_dir, jsons[i]) for i in selc]
            cfg_win.destroy(); root.destroy()
        Button(cfg_win, text="Start", command=on_start).pack(padx=30, pady=12)
    Button(root, text="Select Agent", command=on_select_agent).pack(padx=30, pady=12)
    root.mainloop()
    return selected["agent"], selected["configs"]

def main():
    agent, configs = select_agent_and_configs_gui()
    if not agent or not configs:
        print("No selection. Exiting."); return
    for cfg in configs:
        run_agent(agent, cfg)

if __name__ == "__main__": 
    main()