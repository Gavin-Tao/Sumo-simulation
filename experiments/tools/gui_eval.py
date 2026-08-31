"""Watch a trained Dublin controller in sumo-gui.

Usage (from repo root or experiments/):
    python experiments/tools/gui_eval.py 207        # exp207 best.pth (masked-8std DQN)
    python experiments/tools/gui_eval.py 215        # exp215 latest ckpt (MoE v2 gate)
    python experiments/tools/gui_eval.py 215 --ckpt <path>   # explicit checkpoint

Greedy policy on the training-eval seed (123). GUI runs via TraCI
(sumo-gui), NOT libsumo — start it from a desktop terminal with DISPLAY.
Read-only tool: trains nothing, writes nothing.
"""
import argparse
import functools
import glob
import os
import sys

os.environ["SUMO_RL_LIBSUMO"] = "0"          # GUI needs TraCI, not libsumo
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments"))

import numpy as np
import torch
import yaml

CFGS = {
    "207": "configs/exp207_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095_rfloor.yaml",
    "215": "configs/exp215_dublin11h_531_moe_enum.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=sorted(CFGS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    os.chdir(os.path.join(_REPO, "experiments"))
    cfg = yaml.safe_load(open(CFGS[args.which]))

    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table
    from sumo_rl.agents.dqn_agent_txw import DQN

    obs_class = functools.partial(
        obsmod.PriorityMovementObservationFunction,
        fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
        priority_source=cfg["priority_source"], awt_cap=float(cfg["obs_awt_cap"]),
        awt_basis=cfg["obs_awt_basis"],
        include_downstream=bool(cfg.get("obs_downstream", False)),
        downstream_fields=tuple(cfg.get("obs_downstream_fields", ())),
        include_lane_occ=bool(cfg.get("obs_lane_occ", False)),
        slot_stats=str(cfg.get("obs_slot_stats", "intent")))
    env = SumoEnvironment(
        net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=True,
        num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
        max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
        single_agent=False, yellow_time=cfg["yellow_time"],
        delta_time=cfg["delta_time"],
        reward_fn=make_priority_avg_waiting_reward(
            load_priority_table(cfg["priority_source"])),
        observation_class=obs_class, sumo_seed=cfg["seed"],
        sumo_warnings=False)
    seed = args.seed if args.seed is not None else int(cfg.get("eval_seed", 123))

    def dqn(od, n_act):
        return DQN(starting_state=tuple([0.0] * od), state_space=od,
                   hidden_dim=cfg["hidden_dim"], action_space=n_act,
                   learning_rate=1e-3, gamma=0.95, epsilon=0.0,
                   target_update=10, capacity=100, mini_size=10 ** 9,
                   batch_size=1, eps_start=0, eps_end=0, eps_decay=1,
                   device="cpu")

    if args.which == "207":
        sys.path.insert(0, os.path.join(_REPO, "experiments", "tools", "kan"))
        from extract_dqn8std_targets import load_meta_tables
        ts_mask, std2green, green2std, turnmap, _ = \
            load_meta_tables(cfg["action_meta_file"])
        env.reset(seed)
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.std_action_map = green2std[tid]
            ts.observation_fn.rebind_movements(turnmap[tid])
            ts.observation_space = ts.observation_fn.observation_space()
        states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
        agent = dqn(len(next(iter(states.values()))), 8)
        ck = args.ckpt or sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
        agent.q_net.load_state_dict(torch.load(
            ck, map_location="cpu", weights_only=False)["policy_state_dict"])
        agent.q_net.eval()

        def act(states):
            out = {}
            for t in env.ts_ids:
                x = torch.tensor(np.asarray(states[t], dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    q = agent.q_net(x)[0].numpy()
                a = int(np.where(ts_mask[t], q, -np.inf).argmax())
                out[t] = int(std2green[t][a])
            return out
    else:
        import moe_glue
        moe = moe_glue.load_moe_tables(cfg["moe_meta_file"])
        env.reset(seed)
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.observation_fn.rebind_movements(moe["turnmap"][tid])
            ts.observation_space = ts.observation_fn.observation_space()
        states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
        experts = moe_glue.build_experts(cfg, moe, env)
        agent = dqn(len(next(iter(states.values()))), 6)
        ck = args.ckpt or sorted(glob.glob(
            f"models/{cfg['name']}/*/ckpt_ep*.pth") + glob.glob(
            f"models/{cfg['name']}/*/best.pth"))[-1]
        agent.q_net.load_state_dict(torch.load(
            ck, map_location="cpu", weights_only=False)["policy_state_dict"])
        agent.q_net.eval()

        def act(states):
            out = {}
            for t in env.ts_ids:
                ts = env.traffic_signals[t]
                props, lv = experts.propose(t, ts.sumo, ts.green_phase)
                m = moe_glue.gate_mask(
                    lv, bool(cfg.get("moe_lexicographic", False)),
                    presence=bool(cfg.get("moe_presence_mask", False)))
                k = int(agent.take_action(states[t], mask=m))
                out[t] = int(props[k])
            return out

    print(f"[gui_eval] exp{args.which}  ckpt={ck}  seed={seed} — "
          f"在 GUI 里按开始键(▶), Delay 调 ~50-100ms 便于观察")
    done = {"__all__": False}
    rew = []
    while not done["__all__"]:
        states, r, done, _ = env.step(action=act(states))
        rew.append(float(np.mean([r[t] for t in env.ts_ids])))
    print(f"[gui_eval] episode done, ep_return={sum(rew):.2f}")
    env.close()


if __name__ == "__main__":
    main()
