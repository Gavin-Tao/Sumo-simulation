"""User-authorized minimal repair of 27 terminal bus routes in old southclosed."""
from pathlib import Path
import hashlib,json,re,sys
import xml.etree.ElementTree as ET
from routing import LaneRoutes
from build_net import NETS,WORK,SOURCE,SOURCE_META,REPO

def main():
    path=NETS/'weekday_18h_15_southclosed/bus_weekday_18h.rou.xml'
    ledger=WORK/'original_bus_terminal_repairs.json'
    if ledger.exists():
        old=json.loads(ledger.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==old['new_sha256']
        print('Already repaired',len(old['vehicles']));return
    router=LaneRoutes(SOURCE,SOURCE_META)
    text=path.read_text();original=text;changes=[]
    for v in ET.fromstring(text).findall('vehicle'):
        rr=v.find('route');edges=rr.get('edges').split()
        if router.net.getEdge(edges[-1]).getToNode().getID() not in router.meta:continue
        desc=router.extend(edges,int(v.get('departLane','0')),both=False)
        assert desc['edges'][:len(edges)]==edges
        pattern=r'(<vehicle\b[^>]*\bid="'+re.escape(v.get('id'))+r'"[^>]*>\s*<route\b[^>]*\bedges=")([^\"]*)(")'
        text,n=re.subn(pattern,lambda m:m[1]+' '.join(desc['edges'])+m[3],text,count=1)
        assert n==1,v.get('id')
        changes.append(dict(id=v.get('id'),old=edges,new=desc['edges'],added=desc['edges'][len(edges):]))
    assert len(changes)==27,len(changes)
    (WORK/'before/original_bus_routes_before_terminal_fix.xml').write_text(original)
    path.write_text(text)
    report=dict(file=str(path.relative_to(REPO)),old_sha256=hashlib.sha256(original.encode()).hexdigest(),
                new_sha256=hashlib.sha256(text.encode()).hexdigest(),vehicles=changes,
                scope='Only route edges of 27 buses changed. IDs, types, departure times and lanes unchanged.')
    ledger.write_text(json.dumps(report,indent=2)+'\n')
    print('Repaired',len(changes),'bus routes; suffix lengths',sorted({len(x['added']) for x in changes}))

if __name__=='__main__':main()
