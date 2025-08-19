import os
import sys

# 检查并设置SUMO_HOME环境变量，这对于使用TraCI是必要的
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
prev_vehicles_speeds = {}
new_vehicles_speeds = {}
each_vehicles_speeds = {}

def get_newly_generated_vehicles(prev_step_vehicles, current_step_vehicles):
    return list(set(current_step_vehicles) - set(prev_step_vehicles))

def set_initial_speed_for_new_vehicles(new_vehicles, speed):
    for vehicle_id in new_vehicles:
        # 获取设置前的速度
        prev_speed = traci.vehicle.getSpeed(vehicle_id)
        prev_vehicles_speeds[vehicle_id] = prev_speed
        print("Previous speed of vehicle", vehicle_id, ":", prev_speed)
        # 设置新生成车辆的初始速度为10
        traci.vehicle.setSpeed(vehicle_id, speed)
        # 获取设置后的速度
        new_speed = traci.vehicle.getSpeed(vehicle_id)
        new_vehicles_speeds[vehicle_id] = new_speed
        print("New speed of vehicle", vehicle_id, ":", new_speed)
    
def run_simulation(sumo_cmd):
    traci.start(sumo_cmd)
    step = 0
    prev_step_vehicles = []  # 初始化前一时间步车辆列表为空
    while step < 1000:  # 运行1000个仿真步
        traci.simulationStep()  # 前进到下一个仿真时间步
        step += 1
        current_step_vehicles = traci.vehicle.getIDList()  # 获取当前时间步的车辆列表
        newly_generated_vehicles = get_newly_generated_vehicles(prev_step_vehicles, current_step_vehicles)
        set_initial_speed_for_new_vehicles(newly_generated_vehicles, 20)  # 设置新生成车辆的初始速度为10
        # print("Newly generated vehicles at step", step, ":", newly_generated_vehicles)
        prev_step_vehicles = current_step_vehicles  # 更新前一时间步的车辆列表
        
        all_vehicles_list = traci.vehicle.getIDList()
        for each_vehicle in all_vehicles_list:
            each_vehicle_speed = traci.vehicle.getSpeed(each_vehicle)
            each_vehicles_speeds[each_vehicle] = each_vehicle_speed
            a=1
        
    traci.close()  # 关闭 TraCI
    
cfg_file = "nets/syc/3x3/1000/syc_new_1000.sumocfg"
if __name__ == "__main__":
    # 定义SUMO命令以启动仿真
    sumoBinary = "sumo-gui"  # 或者使用 "sumo-gui" 来代替 "sumo" 以使用图形界面
    sumoCmd = [sumoBinary, "-c", cfg_file, "--step-length", "1.0"]
    # 调用函数运行仿真
    run_simulation(sumoCmd)
