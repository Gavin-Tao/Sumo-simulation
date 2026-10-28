"""Inventory and GUI preview map for the 18-signal College Green variant."""
import json
import review_final_net18 as review
from build_net18_nocg import NET, META, OUT, CG


if __name__ == "__main__":
    review.NET, review.META, review.OUT = NET, META, OUT
    review.main()
    path = OUT / "final_review/final_net_audit.json"
    data = json.loads(path.read_text())
    data["removed_tls"] = CG
    data["limits"][1] = (
        f"Existing non-RL merges/yields and {len(data['retained_short_approaches'])} "
        "approaches below 50 m remain. College Green south is now a yield; "
        "future demand must measure waiting there as well as queues outside RL inlets."
    )
    path.write_text(json.dumps(data, indent=2) + "\n")
