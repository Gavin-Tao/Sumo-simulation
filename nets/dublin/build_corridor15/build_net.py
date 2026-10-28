"""Build a separate corridor variant from the current southclosed net."""
from pathlib import Path
import copy, hashlib, json, math, os, shutil, subprocess, sys
import xml.etree.ElementTree as ET
import sumolib

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[2]
NETS = REPO / 'nets/dublin'
TAG = 'splitslot15_southclosed_corridor'
NET = NETS / f'dublin_8std_{TAG}.net.xml'
META = NETS / f'dublin_8std_{TAG}_meta.json'
SOURCE = NETS / 'dublin_8std_splitslot15_southclosed.net.xml'
SOURCE_META = NETS / 'dublin_8std_splitslot15_southclosed_meta.json'
GEOMETRY_TLS = 'cluster_2498303858_3624079904_6141082777_6141082778_9936781'
SCENES = ['weekday_18h_15_southclosed', 'weekday_pairs50_15_southclosed']
CHAINS = [('43538577#3', '668344588'), ('-668344588', '-43538577#3'),
          ('3791745#0', '3791745#4'), ('-3791745#5', '-3791745#3')]
KEEPS = ['668344588', '-668344588', '3791745#4', '-3791745#3']
RENAMES = {e: keep for chain, keep in zip(CHAINS, KEEPS) for e in chain if e != keep}
CLOSED = {'4401661#0', '43538584'}
REMOVED_NODES = {'26912118', '26911837', '1426049017', '26911836'}
INTERNAL_PAIRS = {tuple(c) for c in CHAINS}

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def run(args):
    p = subprocess.run(list(map(str, args)), cwd=REPO, capture_output=True, text=True)
    with (WORK/'build.log').open('a') as f: f.write(' '.join(map(str,args))+'\n'+p.stdout+p.stderr+'\n')
    if p.returncode: raise RuntimeError(p.stderr)

def compress(edges):
    out=[]
    for e in edges:
        e=RENAMES.get(e,e)
        if not out or e!=out[-1]:out.append(e)
    return out

def snapshot():
    before=WORK/'before'; before.mkdir(exist_ok=True)
    paths=[SOURCE,SOURCE_META]
    for scene in SCENES:
        paths += [p for p in (NETS/scene).iterdir() if p.is_file()]
    hashes={str(p.relative_to(REPO)):digest(p) for p in paths}
    manifest=before/'source_hashes.json'
    if manifest.exists():
        expected=json.loads(manifest.read_text())
        repair=WORK/'original_bus_terminal_repairs.json'
        if repair.exists():
            approved=json.loads(repair.read_text());expected[approved['file']]=approved['new_sha256']
        assert expected==hashes,'Source changed during build'
    else:
        manifest.write_text(json.dumps(hashes,indent=2)+'\n')
        for p in paths:
            dest=before/p.relative_to(NETS);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    return before

def build():
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    before=snapshot(); src=before/SOURCE.name
    net=sumolib.net.readNet(str(src)); prefix=WORK/'original'
    run([sumolib.checkBinary('netconvert'),'-s',src,'--plain-output-prefix',prefix,'--offset.disable-normalization'])
    trees={s:ET.parse(str(prefix)+s) for s in ('.nod.xml','.edg.xml','.con.xml','.tll.xml')}
    nodes={n.get('id'):n for n in trees['.nod.xml'].getroot()}
    edges={e.get('id'):e for e in trees['.edg.xml'].getroot()}
    for chain,keep in zip(CHAINS,KEEPS):
        assert edges[chain[0]].get('to')==edges[chain[1]].get('from')
        assert len({edges[e].get('numLanes') for e in chain})==1
        shape=[]
        for eid in chain:
            for p in net.getEdge(eid).getRawShape():
                if not shape or math.dist(p,shape[-1])>.001:shape.append(p)
        e=edges[keep];e.set('from',net.getEdge(chain[0]).getFromNode().getID())
        e.set('to',net.getEdge(chain[-1]).getToNode().getID())
        e.set('shape',' '.join(f'{x:.6f},{y:.6f}' for x,y in shape))
        for k in ('length','endOffset'):e.attrib.pop(k,None)
        for child in list(e):
            if child.tag=='lane':
                for k in ('shape','length','endOffset'):child.attrib.pop(k,None)
    for eid in CLOSED|set(RENAMES):trees['.edg.xml'].getroot().remove(edges[eid])
    for nid in REMOVED_NODES:trees['.nod.xml'].getroot().remove(nodes[nid])
    # Preserve the existing, conflict-free boundary departure on the western
    # approach. Keep its first 40 m as a source buffer, then observe the rest of
    # the joined corridor. Adding a new inlet across the outlet's north turn
    # would create another non-RL conflict.
    east=edges['668344588']
    pts=[tuple(map(float,p.split(','))) for p in east.get('shape').split()]
    lane_start=net.getLane('43538577#3_0').getShape()[0]
    dx,dy=pts[1][0]-pts[0][0],pts[1][1]-pts[0][1]
    projected=((lane_start[0]-pts[0][0])*dx+(lane_start[1]-pts[0][1])*dy)/math.hypot(dx,dy)
    remaining=projected+40.0; pre=[pts[0]]; post=None
    for i,(a,b) in enumerate(zip(pts,pts[1:])):
        length=math.dist(a,b)
        if remaining<length:
            split=(a[0]+(b[0]-a[0])*remaining/length,a[1]+(b[1]-a[1])*remaining/length)
            pre.append(split);post=[split]+pts[i+1:];break
        pre.append(b);remaining-=length
    assert post is not None
    join='source_join_668344588'; sourceid='source_668344588'
    ET.SubElement(trees['.nod.xml'].getroot(),'node',id=join,x=str(split[0]),y=str(split[1]),type='priority',radius='0')
    buf=copy.deepcopy(east);buf.set('id',sourceid);buf.set('to',join)
    buf.set('shape',' '.join(f'{x:.6f},{y:.6f}' for x,y in pre))
    east.set('from',join);east.set('shape',' '.join(f'{x:.6f},{y:.6f}' for x,y in post))
    trees['.edg.xml'].getroot().append(buf)
    # net.xml does not retain this zero-radius construction setting.
    nodes['source_join_-77616869#2'].set('radius','0')
    # Restore the collinear tangent lost by net.xml's 2-decimal shape rounding.
    protected=net.getEdge('-77616869#2').getRawShape()
    a,b=protected[:2];dx,dy=b[0]-a[0],b[1]-a[1];norm=math.hypot(dx,dy)
    source=edges['source_-77616869#2']
    ps=[tuple(map(float,p.split(','))) for p in source.get('shape').split()]
    ps[-2]=(a[0]-.5*dx/norm,a[1]-.5*dy/norm);ps[-1]=a
    source.set('shape',' '.join(f'{x:.8f},{y:.8f}' for x,y in ps))
    # Correct the misplaced upstream endpoint of the free outlet. The previous
    # endpoint was behind the inlet stop line, causing the internal lane to
    # double back. Do not shorten the controlled inlet as a geometry workaround.
    outlet=edges['357012332#0']
    outlet.set('shape','4968.00,2000.00 4958.06,1999.83')
    for suffix in ('.con.xml','.tll.xml'):
        root=trees[suffix].getroot()
        for c in list(root.findall('connection')):
            fr,to=c.get('from'),c.get('to')
            if fr in CLOSED or to in CLOSED or (fr,to) in INTERNAL_PAIRS:
                root.remove(c);continue
            if fr is not None:c.set('from',RENAMES.get(fr,fr))
            if to is not None:c.set('to',RENAMES.get(to,to))
    for i in range(2):
        ET.SubElement(trees['.con.xml'].getroot(),'connection',{'from':sourceid,'to':'668344588','fromLane':str(i),'toLane':str(i)})
    for suffix,t in trees.items():t.write(WORK/('corridor'+suffix),encoding='UTF-8',xml_declaration=True)
    raw=WORK/'corridor.net.xml'
    run([sumolib.checkBinary('netconvert'),'-n',WORK/'corridor.nod.xml','-e',WORK/'corridor.edg.xml',
         '-x',WORK/'corridor.con.xml','-i',WORK/'corridor.tll.xml','--lefthand','--offset.disable-normalization',
         '--no-turnarounds','--junctions.corner-detail','5','-o',raw])
    run([sys.executable,'-B',REPO/'experiments/tools/dublin/reindex_8std.py',raw,'--split-slots','--tag',TAG])
    oldmeta=json.loads((before/SOURCE_META.name).read_text())['tls'];newmeta=json.loads(META.read_text())['tls']
    assert set(oldmeta)==set(newmeta)
    for tid,t in oldmeta.items():
        if tid==GEOMETRY_TLS:
            oldstates={v['state'] for v in t['actions'].values() if v.get('valid')}
            newstates={v['state'] for v in newmeta[tid]['actions'].values() if v.get('valid')}
            assert oldstates==newstates,(tid,'green behaviors')
            continue
        for key in ('mask','canonical','std_to_green_index','actions'):assert t[key]==newmeta[tid][key],(tid,key)
        expected=copy.deepcopy(t['links'])
        for grp in expected.values():
            for c in grp:
                for key in ('from_edge','to_edge'):c[key]=RENAMES.get(c[key],c[key])
        assert expected==newmeta[tid]['links'],(tid,'links')
    new=sumolib.net.readNet(str(NET));changes=[]
    for tid in oldmeta:
        for e in net.getNode(tid).getIncoming():
            for l in e.getLanes():
                nl=new.getLane(RENAMES.get(e.getID(),e.getID())+'_'+str(l.getIndex()))
                if abs(l.getLength()-nl.getLength())>1e-8:changes.append([l.getID(),l.getLength(),nl.getID(),nl.getLength()])
    report=dict(source=str(SOURCE.relative_to(REPO)),net=str(NET.relative_to(REPO)),meta=str(META.relative_to(REPO)),
                renames=RENAMES,closed=sorted(CLOSED),removed_nodes=sorted(REMOVED_NODES),rl_inlet_changes=changes,
                tls_count=len(newmeta),corridor_actions_unchanged=True,
                geometry_tls=GEOMETRY_TLS,geometry_old_mask=oldmeta[GEOMETRY_TLS]['mask'],
                geometry_new_mask=newmeta[GEOMETRY_TLS]['mask'],geometry_green_states_unchanged=True,
                west_boundary_buffer_length=new.getEdge('source_668344588').getLength())
    (WORK/'net_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
