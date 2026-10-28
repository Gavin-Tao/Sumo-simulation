"""Boundary-to-boundary diagnostics for the College Green non-RL variant."""
import json
import sys
import sumolib
import validate_net18 as diagnostic
import validate_net18_full_routes as full
from build_net18_nocg import NET, META, OUT, CG


def main():
    policy = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 123
    destination = OUT / "validation"
    destination.mkdir(parents=True, exist_ok=True)
    diagnostic.NET, diagnostic.META, diagnostic.OUT = NET, META, destination
    diagnostic.net = sumolib.net.readNet(str(NET))
    diagnostic.meta = json.loads(META.read_text())["tls"]
    diagnostic.NEW = {"659799", "659805", "9101555", CG}
    # The Foster Place join has been removed along with the added north arm.
    # Retain geometric checks at the restored College Green junction instead.
    diagnostic.NORTH_JOIN = CG
    diagnostic.EXPANDED = {"668344588": ("1448436", 47.69)}
    assert len(diagnostic.meta) == 18 and CG not in diagnostic.meta
    full.OUT = destination
    diagnostic.main(policy, seed, steps=720, route_factory=full.make_routes,
                    require_no_lane_changes=True)


if __name__ == "__main__":
    main()
