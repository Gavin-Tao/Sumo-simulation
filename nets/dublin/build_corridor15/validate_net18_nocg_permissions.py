"""Exercise real passenger/bus/emergency classes without changing scene types."""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ["OMP_NUM_THREADS"] = "1"
import libsumo as sumo
import sumolib
from build_net18_nocg import NET, OUT


def main():
    destination = OUT / "validation/explicit_classes"
    destination.mkdir(exist_ok=True)
    source = OUT / "validation/full_routes_fixed_123.rou.xml"
    root = ET.parse(source).getroot()
    classes = {"car": "passenger", "bus": "bus", "ambulance": "emergency"}
    for i, (name, vclass) in enumerate(classes.items()):
        # Equal physical settings; vehicle class is the permission under test.
        root.insert(i, ET.Element("vType", id=name, vClass=vclass,
                                 length="5", width="1.8", height="1.5", minGap="2.5",
                                 maxSpeed="55.55", accel="2.6", decel="4.5", emergencyDecel="9",
                                 sigma="0.5", tau="1", speedFactor="1", speedDev="0",
                                 lcStrategic="-1", lcCooperative="-1", lcSpeedGain="0", lcKeepRight="0"))
    route = destination / "routes.rou.xml"
    ET.indent(root)
    ET.ElementTree(root).write(route, encoding="UTF-8", xml_declaration=True)
    net = sumolib.net.readNet(str(NET), withInternal=True)
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            assert all(lane.allows(c) for c in classes.values()), lane.getID()
    report = dict(planned=len(root.findall("vehicle")), arrived=0, classes=classes,
                  collisions=0, teleports=0, lane_changes=0, all_lane_permissions_checked=True)
    last, arrived = {}, set()
    sumo.start([str(Path(sys.executable).parent / "sumo"), "-n", str(NET), "-r", str(route),
                "--seed", "123", "--time-to-teleport", "-1", "--ignore-junction-blocker", "-1",
                "--collision.check-junctions", "true", "--collision.action", "warn",
                "--error-log", str(destination / "sumo.log"), "--no-step-log", "true"])
    try:
        for name, vclass in classes.items():
            assert sumo.vehicletype.getVehicleClass(name) == vclass
        for _ in range(3600):
            sumo.simulationStep()
            arrived.update(sumo.simulation.getArrivedIDList())
            report["collisions"] += sumo.simulation.getCollidingVehiclesNumber()
            report["teleports"] += sumo.simulation.getStartingTeleportNumber()
            for vehicle in sumo.vehicle.getIDList():
                lane = sumo.vehicle.getLaneID(vehicle)
                prev = last.get(vehicle)
                if prev and not lane.startswith(":") and prev.rsplit("_", 1)[0] == lane.rsplit("_", 1)[0] and prev != lane:
                    report["lane_changes"] += 1
                last[vehicle] = lane
        report.update(arrived=len(arrived), pending=list(sumo.simulation.getPendingVehicles()),
                      remaining=list(sumo.vehicle.getIDList()))
    finally:
        sumo.close()
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["arrived"] == report["planned"], report
    assert not any(report[k] for k in ("collisions", "teleports", "lane_changes", "pending", "remaining")), report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
