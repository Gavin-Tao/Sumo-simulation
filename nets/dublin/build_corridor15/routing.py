"""Lane-continuous routing with explicit network entrances and exits."""
from collections import Counter
from functools import lru_cache
from itertools import islice
import json
import networkx as nx
import sumolib

class LaneRoutes:
    def __init__(self, net_path, meta_path, zonal_sources=False):
        self.net=sumolib.net.readNet(str(net_path));self.meta=json.load(open(meta_path))['tls']
        self.movements={}
        for tid,t in self.meta.items():
            for grp in t['links'].values():
                for c in grp:
                    self.movements[(c['from_edge']+'_'+str(c['from_lane']),c['to_edge']+'_'+str(c['to_lane']))]=(tid,c['approach']+'-'+c['turn'])
        self.g=nx.DiGraph();self.sources=set();self.sinks=set()
        for edge in self.net.getEdges():
            for lane in edge.getLanes():
                if not lane.allows('passenger'):continue
                lid=lane.getID();self.g.add_node(lid)
                boundary=not edge.getIncoming()
                zonal=zonal_sources and edge.getFromNode().getID() not in self.meta
                if (boundary or zonal) and edge.getToNode().getID() not in self.meta and lane.getLength()>=39.99:self.sources.add(lid)
                if not edge.getOutgoing():self.sinks.add(lid)
                for c in lane.getOutgoing():
                    if not c.getToLane().allows('passenger'):continue
                    target=c.getToLane().getID();key=self.movements.get((lid,target))
                    minor=c.getState()=='m' and key is None
                    self.g.add_edge(lid,target,weight=c.getToLane().getLength()+250*bool(key)+5000*minor,minor=minor,movement=key)
        for lid in self.sources:self.g.add_edge('__SOURCE__',lid,weight=self.net.getLane(lid).getLength())
        for lid in self.sinks:self.g.add_edge(lid,'__SINK__',weight=0)

    @lru_cache(None)
    def prefixes(self,lane,k=6):
        try:return list(islice(nx.shortest_simple_paths(self.g,'__SOURCE__',lane,weight='weight'),k))
        except (nx.NetworkXNoPath,nx.NodeNotFound):return []

    @lru_cache(None)
    def suffixes(self,lane,k=6):
        try:return list(islice(nx.shortest_simple_paths(self.g,lane,'__SINK__',weight='weight'),k))
        except (nx.NetworkXNoPath,nx.NodeNotFound):return []

    def describe(self,path):
        path=[p for p in path if not p.startswith('__')]
        edges=[self.net.getLane(p).getEdge().getID() for p in path]
        traversals=[self.movements[(a,b)] for a,b in zip(path,path[1:]) if (a,b) in self.movements]
        minor=sum(self.g[a][b].get('minor',False) for a,b in zip(path,path[1:]))
        return dict(edges=edges,lane=self.net.getLane(path[0]).getIndex(),lanes=path,
                    traversals=traversals,minor=minor,length=sum(self.net.getLane(p).getLength() for p in path))

    def candidates(self,k=6):
        out={};missing=[]
        for (a,b),movement in self.movements.items():
            found=0
            for pre in self.prefixes(a,k):
                for suf in self.suffixes(b,k):
                    desc=self.describe(pre+suf)
                    if len(desc['edges'])!=len(set(desc['edges'])):continue
                    key=(tuple(desc['edges']),desc['lane'])
                    out[key]=desc;found+=1
            if not found:missing.append(movement)
        return list(out.values()),sorted(set(missing))

    def lane_paths(self,edges,lane):
        paths=[[edges[0]+'_'+str(lane)]]
        for a,b in zip(edges,edges[1:]):
            nxt=[]
            for p in paths:
                if p[-1] not in self.g:continue
                for q in self.g.successors(p[-1]):
                    if q.startswith('__'):continue
                    if self.net.getLane(q).getEdge().getID()==b:nxt.append(p+[q])
            paths=nxt
            if not paths:break
        return paths

    def extend(self,edges,lane,both=True):
        choices=[]
        for core in self.lane_paths(edges,lane):
            pres=self.prefixes(core[0],4) if both and core[0] not in self.sources else [['__SOURCE__',core[0]]]
            sufs=self.suffixes(core[-1],4)
            for pre in pres:
                for suf in sufs:
                    path=pre[:-1]+core+suf[1:]
                    desc=self.describe(path)
                    added=len(desc['edges'])-len(edges)
                    extra_rl=len(desc['traversals'])-sum((a,b) in self.movements for a,b in zip(core,core[1:]))
                    choices.append(((desc['minor'],extra_rl,added,desc['length']),desc))
        if not choices:raise ValueError(('Cannot extend route',edges,lane))
        return min(choices,key=lambda x:x[0])[1]
