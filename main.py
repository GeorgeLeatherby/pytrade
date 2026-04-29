import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import json, importlib
import pandas as pd
from tkinter import Tk, Listbox, SINGLE, MULTIPLE, Button, Label, END, Toplevel, messagebox
from src.environment.single_asset_target_pos_drl_trading_env import MarketDataCache

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

def verify_requested_features(df: pd.DataFrame, config: dict) -> None:
    """Ensure all features marked True in config feature dictionaries exist as columns in the loaded dataframe."""
    requested = set()
    for key in ["saa_features", "paa_asset_token_features", "paa_portfolio_token_features"]:
        requested.update(name for name, enabled in config.get(key, {}).items() if enabled)
    missing = requested - set(df.columns)
    if missing:
        raise ValueError(f"Missing requested features in data: {sorted(missing)}")
    print(f"[verify_requested_features] All {len(requested)} requested features are present.")

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

    # Verifiy requested features are available in the cache before running the agent
    verify_requested_features(df, config)

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

def get_saved_models(agent_folder):
    """
    Get list of saved models in agent's saved_models directory.
    Returns list of directory names that contain best_model.zip.
    """
    saved_models_dir = os.path.join(AGENTS_DIR, agent_folder, "saved_models")
    if not os.path.isdir(saved_models_dir):
        return []
    models = []
    for entry in os.listdir(saved_models_dir):
        model_dir = os.path.join(saved_models_dir, entry)
        if os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "best_model.zip")):
            models.append(entry)
    return sorted(models, reverse=True)  # Most recent first

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

def select_resume_training_gui():
    """
    GUI for selecting a saved model and config to continue training.
    Returns (agent_folder, model_dir_name, config_path) or (None, None, None) if cancelled.
    """
    root = Tk(); root.title("Resume Training")
    agents = discover_agents()
    Label(root, text="Agents:").pack(padx=10, pady=5)
    lb = Listbox(root, selectmode=SINGLE, width=40, height=min(15, len(agents)))
    for a in agents: lb.insert(END, a)
    lb.pack(padx=10, pady=5)

    selected = {"agent": None, "model": None, "config": None}

    def on_select_agent():
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("Selection", "Pick an agent.")
            return
        agent = agents[sel[0]]
        selected["agent"] = agent
        
        # Show saved models window
        models = get_saved_models(agent)
        if not models:
            messagebox.showwarning("No Models", f"No saved models found for {agent}")
            return
        
        model_win = Toplevel(root); model_win.title(f"Saved Models for {agent}")
        Label(model_win, text="Saved Models:").pack(padx=10, pady=5)
        model_lb = Listbox(model_win, selectmode=SINGLE, width=60, height=min(15, len(models)))
        for m in models: model_lb.insert(END, m)
        model_lb.pack(padx=10, pady=5)
        
        def on_select_model():
            sel_m = model_lb.curselection()
            if not sel_m:
                messagebox.showwarning("Selection", "Pick a model.")
                return
            model_name = models[sel_m[0]]
            selected["model"] = model_name
            
            # Show config selection window
            cfg_win = Toplevel(model_win); cfg_win.title(f"Config for {agent}")
            agent_dir = os.path.join(AGENTS_DIR, agent)
            jsons = [f for f in os.listdir(agent_dir) if f.lower().endswith(".json")]
            Label(cfg_win, text="Config:").pack(padx=10, pady=5)
            cfg_lb = Listbox(cfg_win, selectmode=SINGLE, width=60, height=min(15, max(1, len(jsons))))
            for jf in jsons: cfg_lb.insert(END, jf)
            cfg_lb.pack(padx=10, pady=5)
            
            def on_confirm():
                sel_c = cfg_lb.curselection()
                if not sel_c:
                    messagebox.showwarning("Selection", "Pick a config.")
                    return
                config_file = jsons[sel_c[0]]
                selected["config"] = os.path.join(agent_dir, config_file)
                cfg_win.destroy(); model_win.destroy(); root.destroy()
            
            Button(cfg_win, text="Confirm", command=on_confirm).pack(padx=30, pady=12)
        
        Button(model_win, text="Select Model", command=on_select_model).pack(padx=30, pady=12)
    
    Button(root, text="Select Agent", command=on_select_agent).pack(padx=30, pady=12)
    root.mainloop()
    return selected["agent"], selected["model"], selected["config"]

def continue_training_agent(agent_folder, model_dir_name, config_path):
    """
    Continue training from a saved model.
    Dynamically loads the agent module and calls its continue_run function.
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

    # Verify requested features are available in the cache before running the agent
    verify_requested_features(df, config)

    # Initialize MarketDataCache container from dataframe
    cache = MarketDataCache.from_dataframe(
        df, config,
        lookback_window=config["environment"]["lookback_window"],
        maybe_provide_sequence=config["environment"].get("maybe_provide_sequence", False)
    )

    # Model path information
    saved_models_dir = os.path.join(agent_dir, "saved_models")
    model_path = os.path.join(saved_models_dir, model_dir_name, "best_model.zip")

    # Run the agent's continue_run function if available
    if hasattr(mod, "continue_run"):
        result = mod.continue_run(cache, config, model_path, saved_models_dir, model_dir_name)
    else:
        raise RuntimeError(f"Agent module {agent_folder} missing a continue_run(cache, config, model_path, saved_models_dir, model_dir_name) function.")
    
    print(f"\n=== Agent: {agent_folder} | Model: {model_dir_name} | Config: {os.path.basename(config_path)} ===")
    print(result)

def main():
    root = Tk()
    root.title("Trading Agent Training Menu")
    Label(root, text="Select Training Mode:", font=("Arial", 12, "bold")).pack(padx=20, pady=10)
    
    def on_train_new():
        root.destroy()
        agent, configs = select_agent_and_configs_gui()
        if not agent or not configs:
            print("No selection. Exiting."); return
        for cfg in configs:
            run_agent(agent, cfg)
    
    def on_continue_training():
        root.destroy()
        agent, model, config = select_resume_training_gui()
        if not agent or not model or not config:
            print("No selection. Exiting."); return
        continue_training_agent(agent, model, config)
    
    Button(root, text="Train New Agent", command=on_train_new, width=30, height=3).pack(padx=20, pady=10)
    Button(root, text="Continue Training", command=on_continue_training, width=30, height=3).pack(padx=20, pady=10)
    
    root.mainloop()

if __name__ == "__main__": 
    main()