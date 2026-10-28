"""Build a candidate independent source inlet from the CURRENT southclosed net.

Never regenerates routes from old snapshots. Publishing is a separate step.
"""
from pathlib import Path
import json,math,subprocess,sys,os
import xml.etree.ElementTree as ET
import sumolib

WORK=Path(__file__).resolve().parent
REPO=WORK.parents[3]
NETS=REPO/'nets/dublin'
SOURCE=NETS/'dublin_8std_splitslot15_southclosed.net.xml'
if (WORK/'before'/SOURCE.name).exists():
    SOURCE=WORK/'before'/SOURCE.name
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')

def run(args):
    p=subprocess.run(list(map(str,args)),capture_output=True,text=True)
    with (WORK/'build.log').open('a') as f:f.write(' '.join(map(str,args))+'\n'+p.stdout+p.stderr+'\n')
    if p.returncode:raise RuntimeError(p.stderr)

net=sumolib.net.readNet(str(SOURCE));prefix=WORK/'current'
run([sumolib.checkBinary('netconvert'),'-s',SOURCE,'--plain-output-prefix',prefix,'--offset.disable-normalization'])
trees={s:ET.parse(str(prefix)+s) for s in ('.nod.xml','.edg.xml','.con.xml','.tll.xml')}
nodes={x.get('id'):x for x in trees['.nod.xml'].getroot()}
edges={x.get('id'):x for x in trees['.edg.xml'].getroot()}
inlet=net.getEdge('-77616869#2');raw=inlet.getRawShape();end=raw[0]
heading=math.atan2(raw[1][1]-raw[0][1],raw[1][0]-raw[0][0])
end_angle=heading-math.pi/2
radius=6.6 # centerline radius; left-hand 3.2 m lane has 5 m driving radius
arc_end=(end[0]-.5*math.cos(heading),end[1]-.5*math.sin(heading))
center=(arc_end[0]-radius*math.cos(end_angle),arc_end[1]-radius*math.sin(end_angle))
arc=[(center[0]+radius*math.cos(end_angle-math.pi/2+i*math.pi/48),
      center[1]+radius*math.sin(end_angle-math.pi/2+i*math.pi/48)) for i in range(25)]
# Include the former source junction's ~14.7 m passage in the upstream buffer.
straight=146.7
start=(arc[0][0]-straight*math.cos(end_angle),arc[0][1]-straight*math.sin(end_angle))
shape=[start]+arc+[end]
join='source_join_-77616869#2'
ET.SubElement(trees['.nod.xml'].getroot(),'node',id=join,x=str(end[0]),y=str(end[1]),type='priority',radius='0')
nodes['source_node_-77616869#2'].set('x',str(start[0]));nodes['source_node_-77616869#2'].set('y',str(start[1]))
source=edges['source_-77616869#2'];source.set('to',join);source.set('spreadType','right')
source.set('shape',' '.join(f'{x:.6f},{y:.6f}' for x,y in shape))
edges['-77616869#2'].set('from',join)
if '--edge-priority' in sys.argv:
    nodes['cluster_389667_5421212978'].set('rightOfWay','edgePriority')
for suffix,t in trees.items():t.write(WORK/('candidate'+suffix),encoding='UTF-8',xml_declaration=True)
run([sumolib.checkBinary('netconvert'),'-n',WORK/'candidate.nod.xml','-e',WORK/'candidate.edg.xml',
     '-x',WORK/'candidate.con.xml','-i',WORK/'candidate.tll.xml','--lefthand','--offset.disable-normalization',
     '--no-turnarounds','--junctions.corner-detail','5','-o',WORK/'candidate.net.xml'])
run([sys.executable,'-B',REPO/'experiments/tools/dublin/reindex_8std.py',WORK/'candidate.net.xml',
     '--split-slots','--tag','splitslot15_southclosed','--out-dir',WORK])
result=sumolib.net.readNet(str(WORK/'dublin_8std_splitslot15_southclosed.net.xml'))
meta_source=WORK/'before/dublin_8std_splitslot15_southclosed_meta.json'
if not meta_source.exists():meta_source=NETS/meta_source.name
meta=json.loads(meta_source.read_text())['tls']
newmeta=json.loads((WORK/'dublin_8std_splitslot15_southclosed_meta.json').read_text())['tls']
diff=[]
for tid in meta:
    for key in ['mask','canonical','std_to_green_index','actions','links']:
        assert meta[tid][key]==newmeta[tid][key],(tid,key)
    for edge in net.getNode(tid).getIncoming():
        for lane in edge.getLanes():
            nl=result.getLane(lane.getID())
            if nl.getLength()!=lane.getLength():diff.append((lane.getID(),lane.getLength(),nl.getLength()))
print('RL inlet length changes',diff)
for eid in ['77616869#0','-77616869#2','source_-77616869#2','8080823#0']:
    e=result.getEdge(eid);print(eid,e.getLength(),e.getFromNode().getID(),e.getToNode().getID())
    for l in e.getLanes():
        for c in l.getOutgoing():print('  ',l.getID(),c.getToLane().getID(),c.getState())
