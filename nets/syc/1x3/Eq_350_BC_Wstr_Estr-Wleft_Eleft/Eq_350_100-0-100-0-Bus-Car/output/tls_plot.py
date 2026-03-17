import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# ===================== 用户参数 =====================
T_END = 1000
MIN_TEXT_WIDTH = 8
BLOCK_LINEWIDTH = 10
FIGSIZE = (16, 6)
# ==================================================


# ===================== 路径 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_FILE = os.path.join(BASE_DIR, "t_tls_program.xml")

if not os.path.exists(XML_FILE):
    raise FileNotFoundError(f"Cannot find tls_program.xml at {XML_FILE}")


# ===================== 解析 tls_program.xml =====================
tree = ET.parse(XML_FILE)
root = tree.getroot()

tl = root.find("tlLogic")
if tl is None:
    raise RuntimeError("No <tlLogic> found")

# phases: [(state, duration)]
phases = []

for phase in tl.findall("phase"):
    state = phase.attrib["state"]
    duration = float(phase.attrib["duration"])
    phases.append((state, duration))

print("Loaded phases (order preserved):")
for s, d in phases:
    print(f"  {s}  duration={d}")


# ===================== 展开到时间轴 =====================
# segments: (state, t_start, t_end, duration)
segments = []

t = 0.0
while t < T_END:
    for state, dur in phases:
        if t >= T_END:
            break
        t_start = t
        t_end = min(t + dur, T_END)
        segments.append((state, t_start, t_end, dur))
        t += dur


# ===================== state → y 轴映射 =====================
unique_states = []
for state, _, _, _ in segments:
    if state not in unique_states:
        unique_states.append(state)

state_to_y = {s: i for i, s in enumerate(unique_states)}
y_to_state = {i: s for s, i in state_to_y.items()}


# ===================== 画 timeline =====================
plt.figure(figsize=FIGSIZE)

for state, t0, t1, dur in segments:
    y = state_to_y[state]

    # block
    plt.hlines(
        y=y,
        xmin=t0,
        xmax=t1,
        linewidth=BLOCK_LINEWIDTH,
        alpha=0.9
    )

    # duration 标注
    if (t1 - t0) >= MIN_TEXT_WIDTH:
        plt.text(
            (t0 + t1) / 2,
            y + 0.12,
            f"{int(dur)}",
            ha="center",
            va="bottom",
            fontsize=8
        )


plt.xlabel("Simulation time (s)")
plt.ylabel("Signal phase state")
plt.title("Traffic light phase timeline grouped by state (0–1000 s)")

plt.yticks(
    ticks=list(y_to_state.keys()),
    labels=list(y_to_state.values())
)

plt.xlim(0, T_END)
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
