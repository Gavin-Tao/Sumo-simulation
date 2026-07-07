import wandb, pandas as pd
api = wandb.Api()
RUNS = {"exp217": ("sumo-rl-1x1","vfojpy4x"), "exp219": ("sumo-rl-1x1","ro0iaxxy"), "exp220": ("sumo-rl-1x1","a9ykq0ym")}
for name,(proj,rid) in RUNS.items():
    run = api.run(f"taoxw19-/{proj}/runs/{rid}")
    rows = list(run.scan_history(page_size=5000))
    df = pd.DataFrame(rows)
    df.to_pickle(f"{name}_pervisit.pkl")
    k = "eval_system/all/avg_stopped_time_per_visit"
    ep = "train/episode"
    nn = df[k].notna() if k in df else pd.Series(False, index=df.index)
    print(name, run.state, "rows", len(df), "eval-metric rows", int(nn.sum()),
          "ep max", int(df[ep].max()) if ep in df else None, flush=True)
