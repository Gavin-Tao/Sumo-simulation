import sumolib

net_file="nets/syc/1x1/syc_4phases.net.xml"
# 加载SUMO网络文件
net = sumolib.net.readNet(net_file)

# 初始化南北方向和东西方向的车道编号列表
south_north_lanes = []
east_west_lanes = []

# 遍历网络中的所有车道
for lane in net.getEdges():
    # 获取车道的ID
    lane_id = lane.getID()
    # 检查车道的连接情况
    incoming_edges = lane.getIncoming()
    outgoing_edges = lane.getOutgoing()
    # 如果车道有来向连接，表示是南北方向
    if incoming_edges and outgoing_edges:
        south_north_lanes.append(lane_id)
    # 如果车道有去向连接，表示是东西方向
    elif incoming_edges or outgoing_edges:
        east_west_lanes.append(lane_id)

# 输出南北方向和东西方向的车道编号列表
print("南北方向车道编号：", south_north_lanes)
print("东西方向车道编号：", east_west_lanes)
