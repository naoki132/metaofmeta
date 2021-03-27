import copy
import math
import pickle
from functools import total_ordering


# from types import List, Dict, Tuple, Node

from typing import List, Dict, Tuple
Node = Tuple[int, int]
Time = float



class PathData:

    def __init__(self, filepath : str):
        with open(filepath, "rb") as f:
            mapper = pickle.load(f)

        self.nodes = mapper.default_targets
        self.home_poses = mapper.starting_point
        self.distance_of = {node_pair:path[1] for node_pair, path in mapper.paths.items()}

        self.nodes_to_index:Dict[Node, int] = {}
        for i, node in enumerate(self.nodes):
            self.nodes_to_index[node] = i + 1


    # Calc distance of two points
    def distance(self, c1:Node, c2:Node) -> float:
        if c1 == c2: return 0.0
        return self.distance_of[(c1, c2)]


    # Calc distance of coords
    def distance_of_nodes(self, nodes:List[Node]) -> float:

        distance = 0.0
        for i in range(len(nodes)-1):
            distance += self.distance(nodes[i], nodes[i+1])

        return distance


class DroneProperty:#スピード、バッテリー、燃費などを定義

    def __init__(self, pathdata:PathData):
        self.pathdata = pathdata
        self.home_pos = pathdata.home_poses[0]
        self.speed = 0.5
        self.battery_capacity = 3000.0
        self.battery_per_distance = 1.0



class Drone:

    def __init__(self, props:DroneProperty):
        self.props = props

        self.pos_history = [self.props.home_pos]
        self.last_pos = self.props.home_pos
        self.total_distance = 0.0
        self.elapsed_time = 0.0
        self.battery_remain = self.props.battery_capacity


    def _distance(self, c1:Node, c2:Node) -> float:
        return self.props.pathdata.distance(c1, c2)


    def move_to(self, target:Node) -> None:
        if self.last_pos == target: return

        c_distance = self._distance(self.last_pos, target)

        self.pos_history.append(target)
        self.last_pos = target

        self.total_distance += c_distance
        self.elapsed_time += c_distance * self.props.speed

        self.battery_remain -= c_distance * self.props.battery_per_distance
        if self.battery_remain < 0:
            raise RuntimeError('Out of battery.')

        if self.last_pos == self.props.home_pos:
            self.battery_remain = self.props.battery_capacity



    def try_move_to(self, target:Node) -> bool:

        if self._distance(self.last_pos, target) + self._distance(target, self.props.home_pos) > self.battery_remain:
            return False

        self.move_to(target)
        return True


    def return_home(self) -> None:
        self.move_to(self.props.home_pos)


class PlanGenerator: #ここで評価値算出してる。firefly.pyの評価値算出のところで呼ばれてた

    def __init__(self, *,
        pathdata:PathData,
        drone_prop:DroneProperty,
        n_drones:int,
        drone_n_cp,
        safety_weight:float, # 不確かさの重み
        distance_weight:float, # 総距離の重み
        e, #遺伝子
    ):
        self.pathdata = pathdata
        self.drone_prop = drone_prop
        self.n_drones = n_drones
        self.drone_n_cp = drone_n_cp
        self.safety_weight = safety_weight
        self.distance_weight = distance_weight
        self.e = e


    def make(self, clusters_nodes, e):
        return Plan(self, clusters_nodes, e) #firefly.pyでcalc_valueが呼ばれると最終的にここが呼ばれる


@total_ordering
class Plan:

    def __init__(self, props:PlanGenerator, clusters_nodes:List[List[Node]], e:List[int]):
        self.clusters_nodes = clusters_nodes
        self.drones = [Drone(props.drone_prop) for _ in range(props.n_drones)] # ドローン台数だけプロパティー用意してる

        nodes_to_index = props.drone_prop.pathdata.nodes_to_index

        last_visit_time_on_nodes = {}
        i_drone = 0
        text = ''
        wariate = [1] * props.n_drones
        j = 0

        for i in range(len(e)):
            if e[i] == 0:
                wariate[j] += 1 # インデックスアウトってエラー出た
            elif e[i] == 1:
                j += 1

        for nodes in self.clusters_nodes:
            
            text += '['
            remain_nodes = copy.copy(nodes)

            while len(remain_nodes): #remain_nodesが空になるまでループ
                drone = self.drones[i_drone]
                text += '|{}>'.format(i_drone)

                while len(remain_nodes): #remain_nodesが空になるor充電切れまでループ
                    node = remain_nodes[0] # nodeに先頭の座標を代入
                    if wariate[i_drone] <= 0: break #割り当て数が0になったらループ終了して次のドローンへ
                    if not drone.try_move_to(node): break #たどりつく前に充電切れたらループ終了して、次のドローンへ
                    last_visit_time_on_nodes[node] = drone.elapsed_time
                    text += '{:>2} '.format(nodes_to_index[node])
                    remain_nodes.pop(0)
                    wariate[i_drone] -= 1 #割り当て数1減らす
                        
                drone.return_home()
                i_drone = (i_drone + 1) % props.n_drones

            text += ']'



        self.total_distance = sum([drone.total_distance for drone in self.drones])
        self.whole_time = max([drone.elapsed_time   for drone in self.drones])
        safety_on_nodes = [math.exp(-0.001279214 * (self.whole_time - last_visit_time)) for last_visit_time in last_visit_time_on_nodes.values()]
        self.average_safety = sum(safety_on_nodes) / len(safety_on_nodes)

    #   normalized_distance = ((sum_distance - distance_range.start) / (distance_range.stop - distance_range.start))
        self.value = props.safety_weight * (1.0 - self.average_safety) + props.distance_weight * self.total_distance
        self.text = text


    def __lt__(self, plan): # less thanの意味で、＜を使うときに呼ばれる
        if not isinstance(plan, Plan): return NotImplemented
        return self.value < plan.value


    def __eq__(self, plan): # =を使うときに呼ばれる
        if not isinstance(plan, Plan): return NotImplemented
        return self.value == plan.value



