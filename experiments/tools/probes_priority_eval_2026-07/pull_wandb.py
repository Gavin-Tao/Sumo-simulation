"""Pull eval_system per-visit series for exp207/211/215 from wandb."""
import pickle
import wandb

RUNS = {"exp207": "m258ma2j", "exp211": "cqmihwfl", "exp215": "4fr5ugh0"}
CLASSES = ["car", "bus", "ambulance", "all"]
KEYS = ["train/episode", "eval_system/mean_reward", "eval_system/pending_veh",
        "eval_system/all/completion_rate"]
for c in CLASSES:
    KEYS += [f"eval_system/{c}/avg_stop_events_per_visit",
             f"eval_system/{c}/avg_stopped_time_per_visit"]

api = wandb.Api()
for name, rid in RUNS.items():
    run = api.run(f"taoxw19-/sumo-rl-dublin/runs/{rid}")
    rows = list(run.scan_history(keys=KEYS, page_size=1000))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_pickle(f"{name}_pervisit.pkl")
    print(name, rid, run.state, "rows:", len(df),
          "ep range:", df["train/episode"].min(), "-", df["train/episode"].max())
