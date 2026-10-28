"""Apply the same independent state/reward diagnostics to the final 19 TLS net."""
import json
import sys
import sumolib
import validate_net18 as diagnostic
import validate_net18_full_routes as full
from build_net19 import NET,META,OUT


def main():
    policy=sys.argv[1] if len(sys.argv)>1 else "fixed"
    seed=int(sys.argv[2]) if len(sys.argv)>2 else 123
    external="--full-routes" in sys.argv
    destination=OUT/"validation"
    if external:destination=destination/"full_routes"
    destination.mkdir(parents=True,exist_ok=True)
    diagnostic.NET=NET;diagnostic.META=META;diagnostic.OUT=destination
    diagnostic.net=sumolib.net.readNet(str(NET));diagnostic.meta=json.loads(META.read_text())["tls"]
    diagnostic.NEW=diagnostic.NEW|{"9101555"}
    assert len(diagnostic.meta)==19
    full.OUT=destination
    diagnostic.main(policy,seed,steps=720 if external else 240,
                    route_factory=full.make_routes if external else None,
                    require_no_lane_changes=True)


if __name__=="__main__":main()
