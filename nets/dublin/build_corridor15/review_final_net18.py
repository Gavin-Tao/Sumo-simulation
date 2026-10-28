"""Final topology inventory and map; does not edit the network or scenarios."""
from pathlib import Path
import hashlib
import json
import sys
import textwrap
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sumolib
from build_net18 import NET,META,REFERENCE,OUT,REPO


def main():
    directory=OUT/"final_review";directory.mkdir(exist_ok=True)
    n=sumolib.net.readNet(str(NET));ref=sumolib.net.readNet(str(REFERENCE))
    meta=json.loads(META.read_text())["tls"]
    assert len(meta) in (18,19) and "133307292" not in meta
    count=len(meta)
    assert {t.getID() for t in n.getTrafficLights()}==set(meta)
    rows=[];short=[];minor=[]
    order=[t for t in meta if t!="9101555"]+(["9101555"] if "9101555" in meta else [])
    for number,tid in enumerate(order,1):
        node=n.getNode(tid);assert ref.getNode(tid).getType()=="traffic_light"
        actual={(c.getFrom().getID(),c.getFromLane().getIndex(),c.getTo().getID(),c.getToLane().getIndex())
                for c in node.getConnections()}
        declared={(c["from_edge"],c["from_lane"],c["to_edge"],c["to_lane"])
                  for group in meta[tid]["links"].values() for c in group}
        assert actual==declared,tid
        greens=[v["state"] for v in meta[tid]["actions"].values() if v.get("valid")]
        assert all("g" not in s for s in greens)
        assert all(any(s[int(index)]=="G" for s in greens) for index in meta[tid]["links"])
        rows.append(dict(number=number,tls=tid,dcc_reference_signal=True,actions=sum(meta[tid]["mask"]),
                         connections=len(actual),position=node.getCoord()))
        for e in node.getIncoming():
            if e.getLength()<50:short.append(dict(tls=tid,edge=e.getID(),length=e.getLength()))
        for e in node.getOutgoing():
            downstream=e.getToNode()
            if downstream.getID() in meta:continue
            for c in downstream.getConnections():
                if c.getFrom()==e and c.getState()=="m":
                    minor.append(dict(tls=tid,outlet=e.getID(),length=e.getLength(),downstream=downstream.getID(),
                                      from_lane=c.getFromLane().getID(),to_lane=c.getToLane().getID()))
    previews=[]
    for name in ("weekday_18h_15_southclosed_corridor","weekday_pairs50_15_southclosed_corridor"):
        p=REPO/"nets/dublin"/name/"net_preview.sumocfg"
        settings=ET.parse(p).getroot()
        assert (p.parent/settings.find("input/net-file").get("value")).resolve()==NET
        assert settings.find("input/route-files") is None
        previews.append(str(p.relative_to(REPO)))
    entries=[e for e in n.getEdges() if not e.getIncoming()]
    assert all(e.getToNode().getID() not in meta for e in entries)
    report=dict(net=str(NET.relative_to(REPO)),meta=str(META.relative_to(REPO)),
                net_sha256=hashlib.sha256(NET.read_bytes()).hexdigest(),
                meta_sha256=hashlib.sha256(META.read_bytes()).hexdigest(),
                tls=rows,total_connections=sum(r["connections"] for r in rows),
                boundary_entry_edges=[e.getID() for e in entries],previews=previews,
                decision_9101555=dict(add="9101555" in meta,replace_existing=False,dcc_signal=True,
                    reason=("Added as a controlled side-road priority and Mercer corridor test point. The 70.85 m outlet's uncontrolled yield at 9101554 remains a mandatory demand-calibration check."
                            if "9101555" in meta else "The 70.85 m outlet ends at uncontrolled yield node 9101554; old traffic volume is not the decision criterion.")),
                retained_short_approaches=short,retained_minor_connections_after_rl_outlets=minor,
                limits=["Topology and diagnostic routes are validated, not future traffic capacity or learned priority performance.",
                        "Existing non-RL merges/yields and five approaches below 50 m remain. Future calibration must measure queues outside RL inlets.",
                        "Formal routes must originate on allowed non-RL entry roads and finish at boundary exits; net topology alone does not enforce vehicle origins."])
    (directory/"final_net_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    fig=plt.figure(figsize=(18,10),facecolor="white")
    ax=fig.add_axes([.035,.08,.64,.83]);panel=fig.add_axes([.70,.07,.29,.84]);panel.axis("off")
    for e in n.getEdges():
        shape=e.getShape();source=e.getID().startswith("source_")
        controlled=e.getToNode().getID() in meta
        color="#d89b23" if source else ("#277da1" if controlled else "#a6adb3")
        ax.plot([p[0] for p in shape],[p[1] for p in shape],color=color,lw=1.8 if source else 1.0,zorder=1)
    for e in entries:
        x,y=e.getShape()[0];ax.scatter(x,y,s=24,c="#d89b23",marker="^",zorder=3)
    for row in rows:
        x,y=row["position"];ax.scatter(x,y,s=220,c="#ba3b46",edgecolors="white",linewidths=.7,zorder=4)
        ax.text(x,y,str(row["number"]),ha="center",va="center",color="white",fontsize=8,fontweight="bold",zorder=5)
    if "9101555" not in meta:
        x,y=n.getNode("9101555").getCoord();ax.scatter(x,y,s=28,c="#555",marker="s",zorder=4)
        ax.annotate("9101555: retained as yield",xy=(x,y),xytext=(x+180,y-85),fontsize=8,color="#444",
                    arrowprops=dict(arrowstyle="-",color="#777",lw=.7),zorder=5)
    ax.set_aspect("equal");ax.set_xlabel("Local SUMO coordinate x (m)");ax.set_ylabel("Local SUMO coordinate y (m)")
    ax.grid(alpha=.13);ax.tick_params(labelsize=8)
    ax.legend(handles=[Line2D([0],[0],color="#277da1",lw=2,label="RL incoming road"),
                       Line2D([0],[0],color="#a6adb3",lw=2,label="Other existing road"),
                       Line2D([0],[0],color="#d89b23",lw=2,label="Artificial entry road")],loc="lower left",fontsize=8)
    panel.text(0,1,f"{count} RL SIGNALS",fontsize=14,weight="bold",va="top")
    y=.955
    for row in rows:
        lines=textwrap.wrap(row["tls"],width=51,break_long_words=True,break_on_hyphens=False)
        text=f"{row['number']:02d}  "+("\n     ".join(lines))+f"\n     {row['actions']} valid actions"
        panel.text(0,y,text,fontsize=8,family="monospace",va="top",linespacing=1.2)
        y-=.031+.013*len(lines)
    fig.suptitle(f"Dublin corridor — final network stage: {count} RL signals",fontsize=17,weight="bold",x=.44,y=.965)
    fig.text(.04,.025,"Topology only. Artificial study boundaries are marked; traffic demand and priority performance are not calibrated.",fontsize=10,color="#555")
    fig.savefig(directory/"final_net_overview.png",dpi=180)
    fig.savefig(directory/"final_net_overview.svg")
    print(json.dumps(dict(tls=len(rows),connections=report["total_connections"],entries=len(entries),
                          short_approaches=short,output=str(directory)),indent=2))


if __name__=="__main__":main()
