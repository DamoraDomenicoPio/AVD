# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import colorama
from colorama import Back, Style
import random
import math
import numpy as np
import carla
from my_utils import *
from misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance,get_stop_sign_trigger_location)
from basic_agent import BasicAgent
from local_planner import RoadOption, LocalPlanner
from behavior_types import Cautious, Aggressive, Normal, MyAgent

from misc import get_speed, positive, is_within_distance, compute_distance

# import os

# pipe_path = '/tmp/custom_pipe'

# # Check if the named pipe exists
# if not os.path.exists(pipe_path):
#     os.mkfifo(pipe_path)

# def custom_print(message):
#     with open(pipe_path, 'w') as pipe:
#         pipe.write(message + '\n')

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='myAgent', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """
        colorama.init()
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

        # parametri Domenico
        self._overtake = False
        self._is_stopped = False
        self._count_is_stopped = 0
        self._is_overtaking = False
        self._begin_overtake = False
        self._end_overtake = False
        self._distance_other_lane = -1
        self._counter_wait_for_overtake = 0
        self._my_lane_id = 0

        #parametri Assia
        self._stop_counter = 0
        self._current_stop_sign = None
        self._done_wating_at_stop = False
        self._stop_timer_time = 0
        self._stop_duration_seconds = 3  # restiamo fermi allo stop per almeno 3 secondi 
        self._current_junction = None
        self._current_straight_yaw = None

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()
        
        elif behavior == 'myAgent':
            self._behavior = MyAgent()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("traffic.traffic_light")
        affected, traffic_light_object = self._affected_by_traffic_light(lights_list,12)

        return affected, traffic_light_object
    
    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    #Ferdinando
    def vehicle_is_a_bicycle(self,vehicle_type_id: str):
        bycicles = ['vehicle.bh.crossbike','vehicle.diamondback.century','vehicle.gazelle.omafiets']
        return vehicle_type_id in bycicles

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        
        bicycle_list= self._world.get_actors().filter("*vehicle*")
        bicycle_list= [bicycle for bicycle in bicycle_list if self.vehicle_is_a_bicycle(bicycle.type_id) and dist(bicycle) < 10]

        #vehicle_list.extend(bicycle_list) #se decommenti le prossime 4 righe, commenta questa

        if len(bicycle_list) == 1:
            print("Bicycle Crossing")
            print(bicycle_list[0].type_id)
            return True, bicycle_list[0], dist(bicycle_list[0])

        self._local_planner.set_lateral_offset(0)
        
        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)  #TODO qui

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]
        print(walker_list)
        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._pedestrian_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._pedestrian_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._pedestrian_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance
        
    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            # target_speed = min([
            #     positive(vehicle_speed - self._behavior.speed_decrease),
            #     self._behavior.max_speed,
            #     self._speed_limit - self._behavior.speed_lim_dist])
            # self._local_planner.set_speed(target_speed)
            # control = self.slow_down(6,flag=False)
            # control = self._local_planner.run_step(debug=debug)



            control = self.slow_down(6,flag=False)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control
    

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        # custom_print('\n\nEmergency stop\n\n')
        return control

    #Assia
    def smooth_accelerate(self):
        control = carla.VehicleControl()
        control.throttle = 0.5
        control.brake = 0.0
        return control
    
    #Assia
    def smooth_stop(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.25
        return control 

    def traffic_stop_manager(self, max_distance=8):
        """
        This method is in charge for detecting stop signals.
        """
        actor_list = self._world.get_actors()
        stops_list = actor_list.filter("traffic.stop")
        affected, stop_object = self._affected_by_stop(stops_list, max_distance)

        return affected,stop_object

    #Ferdinando
    def get_upcoming_waypoints(self, distance=5):
        map = self._world.get_map()
        current_waypoint = map.get_waypoint(self._vehicle.get_location())
        upcoming_waypoints=current_waypoint.next(distance) 
        
        return upcoming_waypoints
    
    #Ferdinando
    def is_waypoint_occupied(self, waypoint,max_distance=5):
        vehicles_list = self._world.get_actors().filter("vehicle.*")

        for vehicle in vehicles_list:
            vehicle_location=vehicle.get_location()
            distance = math.sqrt((vehicle_location.x - waypoint.transform.location.x)**2 +
                             (vehicle_location.y - waypoint.transform.location.y)**2 +
                             (vehicle_location.z - waypoint.transform.location.z)**2)
            if distance <= max_distance:
                return True
        return False
    
    #Ferdinando
    def get_upcoming_waypoints_occupied(self):
        upcoming_waypoints = self.get_upcoming_waypoints()
        
        occupied_waypoints = []
        for waypoint in upcoming_waypoints:
            if self.is_waypoint_occupied(waypoint):
                occupied_waypoints.append(waypoint)
        
        return occupied_waypoints
    
    #Ferdinando
    def slow_down(self,speed_decrease,flag=True):
        """La macchina rallenta alla velocità speed decrease"""
        if flag:
            target_speed = speed_decrease
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step()
            return control
        else:
            vehicle_speed = get_speed(self._vehicle)
            target_speed = vehicle_speed - speed_decrease
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step()
            return control

    #Ferdinando
    def set_behavior(self,behavior):
        """Set the vehicle behavior"""

        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    #Ferdinando
    def get_front_of_vehicle(self,other_vehicle):
        # Get the forward vector from the reference transform
        fwd=other_vehicle.get_transform().get_forward_vector() 
        forward_vector = np.array([fwd.x,fwd.y,fwd.z])
        
        # Calculate the front location
        other_vehicle_bb=other_vehicle.bounding_box 
        other_vehicle_location=other_vehicle_bb.location
        front_location = np.array([other_vehicle_location.x,other_vehicle_location.y,other_vehicle_location.z]) + other_vehicle_bb.extent.x * forward_vector 

        # Return the front location as a Location object
        return carla.Location(front_location[0], front_location[1], front_location[2])
    
    #Ferdinando
    def vehicle_checkout_for_intersections(self,other_vehicle,distance,angle_list=None):
        waypoint_front_of_vehicle=self._map.get_waypoint(self.get_front_of_vehicle(other_vehicle))
        if is_within_distance(waypoint_front_of_vehicle.transform, self._vehicle.get_transform(), distance, angle_list):
            return True
        else: return False

    #Ferdinando
    def vehicle_stopped_list_counter(self,lst):
        # Trova l'ultimo elemento pieno della lista
        last_filled_index = None
        for i in range(len(lst)-1, -1, -1):
            if lst[i] is not None:
                last_filled_index = i
                break
        
        # Se la lista è completamente piena e tutti gli elementi sono uguali, ritorna True e una lista vuota
        if last_filled_index == len(lst) - 1 and all(lst[i] == lst[0] for i in range(1, len(lst))):
            return True, [None] * len(lst)

        # Controlla se l'ultimo elemento pieno è uguale al precedente
        if last_filled_index > 0 and lst[last_filled_index] != lst[last_filled_index - 1]:
            last_filled_element = lst[last_filled_index]
            # Elimina tutti gli elementi precedenti e metti l'ultimo elemento pieno come primo
            lst = [last_filled_element] + [None] * (len(lst) - 1)

        return False, lst

    #NOTE Assia
    def wait_at_stop(self):
        '''
            Metodo per fare in modo che l'agente attenda un certo numero di secondi
            di simulazione allo stop. Se ha aspettato abbastanza ritorna False (non deve aspettare) 
            altrimenti ritorna True (deve continuare ad aspettare)
        '''
        if not self._done_wating_at_stop:
            current_simulation_time = self._world.get_snapshot().timestamp.elapsed_seconds
            if current_simulation_time - self._stop_timer_time < self._stop_duration_seconds:
                self._done_wating_at_stop = False
                return True
            else:
                self._done_wating_at_stop = True
                self._stop_timer_time = None
                return False
    
    #NOTE Assia
    def start_stop_timer(self): 
        self._stop_timer_time = self._world.get_snapshot().timestamp.elapsed_seconds
    

    # ----------- INCROCI ASSIA------------------

    #NOTE Assia
    def get_vehicles_in_junction(self, junction): 
        '''
        Ritorna la lista di veicoli (escluso l'ego) che sono nei pressi di un incrocio
        '''
        vehicles = self._world.get_actors().filter('vehicle.*')

        j_bbox = junction.bounding_box
        j_center = j_bbox.location  # il centro dell'incrocio
        dim_x = j_bbox.extent.x*2
        dim_y = j_bbox.extent.y*2
        # check_distance è la distanza entro la quale un'auto è considerata all'interno dell'incrocio
        check_distence = max(dim_x, dim_y) + 4   # è la dimenzione più grande più 4 metri (più o meno la dimensione di una macchia)

        vehicles_in_j = []
        for v in vehicles:
            if v.id != self._vehicle.id:
                # Se il veicolo è entro la check_distance...
                if j_center.distance(v.get_location()) < check_distence:
                    vehicles_in_j.append(v) # viene aggiunto alla lista

        return vehicles_in_j

    #TODO Assia
    def create_junction_dict(self, junction): 
        '''
        Ritorna un dizionario con informazioni sui veicoli all'incriocio.
        La chiave è l'id del veicolo
        Intention è la strada su cui vuole andare (rispetto all'ego vehicle) [STRAIGHT, LEFT, RIGHT]
        Speed è la velocità attuale
        '''
        vehicles = self.get_vehicles_in_junction(junction)
        return 
    
    #TODO Assia
    def get_junction_occupiers(self, junction): 
        vehicles = self.get_vehicles_in_junction(junction)
        occupiers = []
        for v in vehicles:
            if v.id != self._vehicle.id: 
                if self._world.get_map().get_waypoint(v.get_location()).is_junction:
                    velocity = v.get_velocity()
                    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                    if speed > 3:  #TODO metter e a zero?
                        occupiers.append(v)

        return occupiers
    
    #TODO Assia
    def is_yelding(self, vehicle):
        is_free = True
        if self._world.get_map().get_waypoint(vehicle.get_location()).is_junction:
                    velocity = vehicle.get_velocity()
                    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                    if speed > 0.5:
                        is_free = False

        return is_free

    #Assia
    def get_angle_between_vehicles(self, yaw1, yaw2):
        ego_dir = get_forward(yaw1)
        other_dir = get_forward(yaw2)
        alfa = get_angle_between_vectors(ego_dir, other_dir)
        return alfa

    #Assia
    def goes_straight(self, vehicle, threshold=15):
        # ego_yaw = self._world.get_map().get_waypoint(self._vehicle.get_location()).transform.rotation.yaw
        ego_yaw = self._current_straight_yaw
        vehicle_yaw = self._world.get_map().get_waypoint(vehicle.get_location()).transform.rotation.yaw

        is_free = True
        alfa = self.get_angle_between_vehicles(ego_yaw, vehicle_yaw)
        print(Back.MAGENTA+"Angolo = "+str(alfa)+Style.RESET_ALL+'\n')
        if not (alfa > (180-threshold) and alfa < (180+threshold)): # or (alfa > threshold or alfa < (360-threshold)):
            is_free = False

        if is_free == True: 
            print(Back.GREEN+'\n'+Style.RESET_ALL)
        else: 
            print(Back.RED+'\n'+Style.RESET_ALL)

        return is_free

    #Assia
    def dangerous_trajectory(self, vehicle, threshold=5):
        # ego_yaw = self._world.get_map().get_waypoint(self._vehicle.get_location()).transform.rotation.yaw
        ego_yaw = self._current_straight_yaw
        vehicle_yaw = self._world.get_map().get_waypoint(vehicle.get_location()).transform.rotation.yaw

        dangerous = False
        alfa = self.get_angle_between_vehicles(ego_yaw, vehicle_yaw)
        print(Back.MAGENTA+"Angolo = "+str(alfa)+Style.RESET_ALL+'\n')
        if not (alfa < (180-threshold) or alfa > (90+threshold)): # or (alfa > threshold or alfa < (360-threshold)):
            dangerous = True

        if dangerous == True: 
            print(Back.RED+'\n'+Style.RESET_ALL)
        else: 
            print(Back.GREEN+'\n'+Style.RESET_ALL)

        return dangerous

    #Assia
    def junction_is_free_old(self, junction): 
        '''
        Ritorna vero se l'incrocio è vuoto o gli altri veicoli ci stanno dando la precedenza 
        (cioè se sono fermi)
        '''
        vehicles = self.get_vehicles_in_junction(junction)
        is_free = True
        for v in vehicles:
            if v.id != self._vehicle.id: 
                if self._world.get_map().get_waypoint(v.get_location()).is_junction:
                    velocity = v.get_velocity()
                    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                    if speed > 0:
                        is_free = False

        return is_free

    #Assia
    def junction_is_free(self, junction):
        vehicles = self.get_vehicles_in_junction(junction)
        is_free = True
        occupiers = self.get_junction_occupiers(junction)
        for o in occupiers:
            if self._incoming_direction ==  RoadOption.STRAIGHT:
                if not self.goes_straight(o):
                    is_free = False
            elif self._incoming_direction ==  RoadOption.RIGHT:
                if self.dangerous_trajectory(o):
                    is_free = False
            else:
                if not self.is_yelding(o):
                    is_free = False
        return is_free

    #Ferdinando
    def is_emergency_vehicle(self,vehicle):
        return 'ambulance' in vehicle.type_id or 'police' in vehicle.type_id 
    
    #Ferdinando
    def get_distance(self,location1, location2):
        dx = location1.x - location2.x
        dy = location1.y - location2.y
        dz = location1.z - location2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    #Ferdinando
    def dot_product(self, location1, location2):
        # Estrai le coordinate dei due waypoint
        x1, y1, z1 = location1.x, location1.y, location1.z
        x2, y2, z2 = location2.x, location2.y, location2.z
        
        # Calcola il prodotto scalare
        scalar_product = x1 * x2 + y1 * y2 + z1 * z2
        
        return scalar_product

    #Ferdinando
    def emergency_vehicle_case_actuation(self,other_vehicle,break_distance=12):
        if self.is_emergency_vehicle(other_vehicle):
            ego_vehicle_wp=self._map.get_waypoint(self._vehicle.get_location())
            other_vehicle_wp=self._map.get_waypoint(other_vehicle.get_location())
            if ego_vehicle_wp.road_id == other_vehicle_wp.road_id and self.dot_product(self._vehicle.get_location(),other_vehicle.get_location()) > 0:
                print("Il veicolo di emergenza guida nel mio stesso senso di marcia")
                if ego_vehicle_wp.lane_id == other_vehicle_wp.lane_id:
                    print("Il veicolo di emergenza è sulla mia stessa corsia")
                    start_lane_wp=ego_vehicle_wp.previous_until_lane_start(1)[-1]
                    distance_among_ego_and_start= self.get_distance(ego_vehicle_wp.transform.location,start_lane_wp.transform.location)
                    distance_among_other_and_start= self.get_distance(other_vehicle_wp.transform.location,start_lane_wp.transform.location)
                    if distance_among_ego_and_start > distance_among_other_and_start and self.get_distance(self._vehicle.get_location(),other_vehicle.get_location()) < break_distance:
                        print("Il veicolo di emergenza",other_vehicle.type_id,"è DIETRO la mia macchina e a ",break_distance," metri da me!")
                        return True
                    else: 
                        print("NO ACTUATION 1") # Non attuare il controllo
                        return False
                else:
                    print("Il veicolo di emergenza è su un'altra corsia")
                    start_ego_lane_wp=ego_vehicle_wp.previous_until_lane_start(1)[-1]
                    start_other_vehicle_lane_wp=other_vehicle_wp.previous_until_lane_start(1)[-1]
                    distance_among_ego_and_ego_start= self.get_distance(ego_vehicle_wp.transform.location,start_ego_lane_wp.transform.location)
                    distance_among_other_and_other_start= self.get_distance(other_vehicle_wp.transform.location,start_other_vehicle_lane_wp.transform.location)
                    if distance_among_ego_and_ego_start > distance_among_other_and_other_start and self.get_distance(self._vehicle.get_location(),other_vehicle.get_location()) < break_distance:
                        print("Il veicolo di emergenza",other_vehicle.type_id,"è DIETRO la mia macchina e a ",break_distance," metri da me!")
                        return True
                    else: 
                        print("NO ACTUATION 2") # Non attuare il controllo
                        return False
            else:
                print("Il veicolo di emergenza guida nel senso di marcia contrario al mio")
                if ego_vehicle_wp.lane_id != other_vehicle_wp.lane_id and self.get_distance(self._vehicle.get_location(),other_vehicle.get_location()) < break_distance:
                    print("Il veicolo di emergenza",other_vehicle.type_id,"è DAVANTI la mia macchina su un altra corsia e a ",break_distance," metri da me!")
                    return True
                else:
                    print("NO ACTUATION 3") # Non attuare il controllo
                    return False #il veicolo sta nella mia stessa corsia con verso opposto e sta davanti a me oppure nell'altra corsia ma non è ancora troppo vicino da accostarmi a destra
        else:
            print("Non è un veicolo d'emergenza")    
  
    #Ferdinando 
    def get_angle_between_vehicles2(self, vehicle2):
        ego_dir = self._vehicle.get_transform().get_forward_vector()
        other_dir = vehicle2.get_transform().get_forward_vector()
        print("FW_EGO FW_OTHER",ego_dir,other_dir)
        alfa = get_angle_between_vectors(ego_dir, other_dir)
        return alfa    

    #Ferdinando
    def get_vehicle_displacement(self,vehicle):
        """
        Calcola lo scostamento del veicolo rispetto al centro della corsia.
        """
        vehicle_location = vehicle.get_location()
        ego_vehicle_location = self._vehicle.get_location()
        print("LOC X:",ego_vehicle_location.x,vehicle_location.x)
        print("LOC Y:",ego_vehicle_location.y,vehicle_location.y)
        print("LOC Z:",ego_vehicle_location.z,vehicle_location.z)
        
        if ego_vehicle_location.y > vehicle_location.y:
            direction = 'left'
        elif ego_vehicle_location.y < vehicle_location.y:
            direction = 'right'
        else:
            direction = 'center'
        
        return abs(ego_vehicle_location-vehicle_location),direction

    #Ferdinando
    def offset_actuation(self,vehicle):
        """
        Determina se l'ego vehicle ha spazio sufficiente per proseguire sulla sua corsia e applica l'offset se necessario.
        """
        ego_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        ego_yaw= ego_waypoint.transform.rotation.yaw

        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(vehicle_location)
        vehicle_yaw= vehicle_waypoint.transform.rotation.yaw
        angle=self.get_angle_between_vehicles(ego_yaw,vehicle_yaw)
        print("ANGOLOOO: ", angle)

        if not self.vehicle_is_a_bicycle(vehicle.type_id):
            return False

        # Caso 1: Veicoli (ciclisti) nella stessa corsia
        if self.get_angle_between_vehicles(ego_yaw,vehicle_yaw) < 15:
            print(vehicle_waypoint.lane_id,ego_waypoint.lane_id)
            displacement, direction = self.get_vehicle_displacement(vehicle)
            if direction == 'right' and displacement <= 1:
                print("VEICOLO A DESTRA")
                # Applica un offset verso sinistra per evitare i ciclisti
                self._local_planner.set_lateral_offset(-displacement)
                return True
            elif direction == 'left' or direction == 'center':
                print("VEICOLO A SINISTRA O AL CENTRO: ",direction)
                self._local_planner.set_lateral_offset(0)
                # Non può sorpassare a destra, quindi deve fermarsi
                return False
        else:
            self._local_planner.set_lateral_offset(0)
            return False
        """
        # Caso 2: Veicoli contromano nella corsia adiacente sulla sinistra
        elif self.are_vehicles_traveling_in_opposite_directions(self._vehicle, vehicle):
            print(vehicle_waypoint.lane_id,ego_waypoint.lane_id)
            displacement, direction = self.get_vehicle_displacement(vehicle, ego_waypoint)
            if direction == 'left' and displacement < 1:
                # Applica un offset verso destra per evitare i veicoli contromano
                right_waypoint = self._map.get_waypoint_xodr(
                    ego_waypoint.road_id, ego_waypoint.section_id, ego_waypoint.lane_id + 1
                )
                if right_waypoint and right_waypoint.lane_type == carla.LaneType.Driving:
                    print("VEICOLO SULLA CORSIA ADIACENTE SULLA MIA CORSIA")
                    self._local_planner.set_lateral_offset(+displacement)
                    return True
                else:
                    print("VEICOLO ADIACENTE NON SULLA MIA CORSIA")
                    self._local_planner.set_lateral_offset(0)
                    return False
                """
        #print("CONTINUO NORMALE, NESSUNC CASO PARTICOLARE")      
        #self._local_planner.set_lateral_offset(0)
        #return False

    #Ferdinando
    def offset_actuation_simple(self,vehicle):
        """
        Determina se l'ego vehicle ha spazio sufficiente per proseguire sulla sua corsia e applica l'offset se necessario.
        """

        if not self.vehicle_is_a_bicycle(vehicle.type_id):
            return False

        ego_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        ego_yaw= ego_waypoint.transform.rotation.yaw

        vehicle_waypoint = self._map.get_waypoint(vehicle.get_location())
        vehicle_yaw= vehicle_waypoint.transform.rotation.yaw

        print("EGO YAW - VEHCILE YAW:",ego_yaw,vehicle_yaw)

        #angle=self.get_angle_between_vehicles(ego_yaw,vehicle_yaw)
        angle=self.get_angle_between_vehicles2(vehicle)
        print("ANGOLOOO: ", angle)

        # Caso 1: Veicoli (ciclisti) nella stessa corsia
        if angle < 5:
            # Applica un offset verso sinistra per evitare i ciclisti
            self._local_planner.set_lateral_offset(-1)
            print("BICIIIIIIIIIIIIIII")
            return True
        else:
            return False

    #Ferdinando-Assia  
    def overtake_bike(self,cyclist):
        ego_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        ego_yaw= ego_waypoint.transform.rotation.yaw

        vehicle_waypoint = self._map.get_waypoint(cyclist.get_location())
        vehicle_yaw= vehicle_waypoint.transform.rotation.yaw

        print("EGO YAW - VEHCILE YAW:",ego_yaw,vehicle_yaw)

        #angle=self.get_angle_between_vehicles(ego_yaw,vehicle_yaw)
        angle=self.get_angle_between_vehicles2(cyclist)
        print("ANGOLOOO: ", angle)

        if angle < 10: return True #devo sorpassare
        else: False #non devo sorpassare


# -----------------------------------------------------------------------------------------

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()
        self.set_behavior("normal")

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        ego_transform = self._vehicle.get_transform()
        ego_vehicle_transform_wp = self._map.get_waypoint(ego_transform.location)
        print("Speed: ", get_speed(self._vehicle))
        
        # 1: Red lights and stops behavior
        flag_tf, tf= self.traffic_light_manager()
        if flag_tf:
            self.set_behavior("cautious")
            trigger_location = get_trafficlight_trigger_location(tf)
            trigger_wp = self._map.get_waypoint(trigger_location)
            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), 3, [0, 90]):
                return self.slow_down(0)
        
        # Stop Assia
        stop_affected, ss = self.traffic_stop_manager(3)
        if stop_affected:
            print(Back.CYAN+"Stop"+Style.RESET_ALL)
            print(Back.RED+"Stop affected"+Style.RESET_ALL)
            if self._current_stop_sign == None or self._current_stop_sign.id != ss.id:  # nuovo segnale
                self._done_wating_at_stop = False
            if get_speed(self._vehicle) == 0:
                if self._current_stop_sign == None or self._current_stop_sign.id != ss.id:  # nuovo segnale
                    print(Back.YELLOW+"\n\nINIZIO AD ASPETTARE\n\n"+Style.RESET_ALL)
                    self.start_stop_timer()
                    self._current_stop_sign = ss
                    return self.emergency_stop()
                elif self.wait_at_stop():  # se il segnale era vecchio e devo ancora aspettare
                    print(Back.YELLOW+"ASPETTANDO"+Style.RESET_ALL)
                    return self.emergency_stop()
                else:
                    print(Back.GREEN+"LIBERO"+Style.RESET_ALL)
            else:
                if not self._done_wating_at_stop:
                    print(Back.GREEN+"ASPETTANDO"+Style.RESET_ALL)
                    return self.emergency_stop()
            
            ###
        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            print(Back.CYAN+"Pedestrian avoidance"+Style.RESET_ALL)
            self.set_behavior("cautious")
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance_w = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance_w < self._behavior.braking_distance and get_speed(walker) != 0:
                return self.emergency_stop()


        # 2.2: Car following behaviors
        vehicle_state_2, vehicle_2, distance_2 = self.collision_and_car_avoid_manager(ego_vehicle_wp)


        ##################################### OVERTAKE

        w = self._map.get_waypoint(self._vehicle.get_location())

        for i in range(10):
            print("INCROCIO", w.next(self._sampling_resolution)[0].is_junction)
            w = w.next(self._sampling_resolution)[0]


        obstacles_list = self._world.get_actors().filter("*static.prop*")



        
        obstacles = []
        for o in obstacles_list:
            if o.type_id == "static.prop.constructioncone" or o.type_id == "static.prop.trafficwarning":
                print(o.type_id)
                obstacles.append(o)

        print(len(obstacles), len(obstacles_list))
        
        if len(obstacles) != 0:
            check_obstacle, obstacle, distance_obs =check_vehicle_in_front(self._world, self._map, self._vehicle, distance=45, vehicles=obstacles, check="forward")
            print("OBSTACLE:", check_obstacle, obstacle, distance_obs)
        else:
            check_obstacle, obstacle, distance_obs = False, None, -1

        print("OBSTACLE:", check_obstacle, obstacle, distance_obs)

        
        print("LANE ID:", ego_vehicle_wp.lane_id)

        if not self._overtake and not self._is_overtaking and not self._begin_overtake:
            self._my_lane_id = ego_vehicle_wp.lane_id


        vehicle_state_forward, vehicle_forward, distance_forward = check_vehicle_in_front(self._world, self._map, self._vehicle, distance=45, check="forward")

        if check_obstacle:
            if distance_forward < 0 or distance_forward > distance_obs:
                print("DEVO GESTIRE L'OSTACOLO")
                vehicle_state_2, vehicle_2, distance_2 = check_obstacle, obstacle, distance_obs
            else:
                print("NON NON NON DEVO GESTIRE L'OSTACOLO")
                obstacles = []
        else:
            print("NON NON NON DEVO GESTIRE L'OSTACOLO")
            obstacles = []


        print("DISTANCE OTHER LANE:", self._distance_other_lane)


        vehicle_state, vehicle, distance = vehicle_state_2, vehicle_2, distance_2
        if (vehicle_2 != None and not self.vehicle_is_a_bicycle(vehicle_2.type_id) and (vehicle_state_forward or check_obstacle)) or (self._overtake or self._is_overtaking or self._begin_overtake):
            print("INIZIO OVERTAKE")
            if not self._begin_overtake and not self._is_overtaking:
                if check_overtake(self._world, self._map, self._vehicle, max_distance=30, distance_tail=15, vehicles=obstacles):
                    print("**************")
                    print("**************")
                    print("PIANIFICAZIONE SORPASSO")
                    print("**************")
                    print("**************")
                    self._overtake = True

                else:
                    self._overtake = False

            if self._overtake and not self._begin_overtake:
                waiting , distance_other_lane = wait_overtaking(self._world, self._map, self._vehicle, max_distance=30, distance_tail=15, begin_distance=15, vehicles=obstacles)
                if waiting:
                    print("**************")
                    print("**************")
                    print("ATTENDO IL SORPASSO")
                    print("**************")
                    print("**************")
                    # vehicle_state, vehicle, distance = check_vehicle_in_front(self._world, self._map, self._vehicle, lane_offset=-2, distance=self._distance_other_lane*1.9, check="forward")
                else:
                    print("--------------")
                    print("--------------")
                    print("INIZIO SORPASSO")
                    print("--------------")
                    print("--------------")
                    
                    self.overtake_in_meter(direction="left", distance_same_lane=0,  distance_other_lane=distance_other_lane+1, lane_change_distance=3)
                    self._begin_overtake = True
                    self._is_overtake = True
                    self._distance_other_lane = distance_other_lane
            elif self._overtake and self._begin_overtake:
                vehicle_state, vehicle, distance = check_vehicle_in_front(self._world, self._map, self._vehicle, lane_offset=-2, distance=self._distance_other_lane*2.85, check="forward", angle=[0, 90])

                print("DISTANZA VEICOLO - - - - - - - -", vehicle_state, vehicle, distance)

                control = self._local_planner.run_step()
                if vehicle_state:
                    print("NU ME MOVO . . . . . . . . . . . . . . . . . . . . . . . . .")
                    return self.emergency_stop()
                
                else:
                    print("DAJE ROMA DAJE . . . . . . . . . . . . . . . . . . . . . . . . .")
                    # vehicle_list = self._world.get_actors().filter("*vehicle*")
                    # print("**************")
                    # print("**************")
                    # for v in vehicle_list:
                    #     if self._map.get_waypoint(v.get_location()).lane_id == -1:
                    #         print(v.id, compute_distance(self._vehicle.get_location(), v.get_location()))
                    control = self._local_planner.run_step()
                    self._local_planner.set_speed(15)
                    self._overtake = False
                    return control
            elif not self._overtake and self._begin_overtake:     
                print("STO CAMBIANDO CORSIA")       
                control = self._local_planner.run_step()
                self._local_planner.set_speed(self._speed_limit)
                    
                if ego_vehicle_transform_wp.lane_id * self._my_lane_id < 0:
                    self._is_overtaking = True
                    self._begin_overtake = False
                return control
            elif not self._overtake and not self._begin_overtake and self._is_overtaking:
                vehicle_lane_direction = get_vehicle_lane_direction(self._map, self._vehicle, opposite=True)
                print("ANGOLO:", vehicle_lane_direction)
                if ego_vehicle_transform_wp.lane_id * self._my_lane_id > 0:
                    control = self._local_planner.run_step()
                    self._local_planner.set_speed(10)
                    self._is_overtaking = False
                    return control
                else:


                    # if vehicle_lane_direction >= 15:
                    #     if check_to_right(self._world, self._map, self._vehicle, distance=20, vehicles=obstacles):
                    #         self.end_overtake_in_meter()
                    #         control = self._local_planner.run_step()
                    #         self._local_planner.set_speed(10)
                    #         return control


                    # vehicle_state, vehicle, distance = check_vehicle_in_front(self._world, self._map, self._vehicle, lane_offset=0, distance=60, vehicles=obstacles, check="forward")
                    # vehicle_lane_direction = get_vehicle_lane_direction(self._map, self._vehicle, opposite=True)
                    
                    # print("VEDO A DESTRA", vehicle_lane_direction)
                    # if vehicle_lane_direction >= 355 or vehicle_lane_direction <= 5:
                    #     if check_to_right(self._world, self._map, self._vehicle, distance=20, vehicles=obstacles):
                    #         self.end_overtake_in_meter()
                    #         control = self._local_planner.run_step()
                    #         self._local_planner.set_speed(10)
                    #         # self._is_overtaking = False
                    #         return control
                    # else:
                    control = self._local_planner.run_step()
                    self._local_planner.set_speed(self._speed_limit)
                    self._overtake = False
                    return control

        #####################################


        print("********* SONO FUOIR L'OVERTAKE **************")




        vehicle_state_2, vehicle_2, distance_2 = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        if vehicle_state_2:
            print(Back.CYAN+"Car following"+Style.RESET_ALL)
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance_2 = distance_2 - max(
                vehicle_2.bounding_box.extent.y, vehicle_2.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
            offset_actuation_simple = self.offset_actuation_simple(vehicle_2)
            print("METODOOOOOOOOO FLAG: ",offset_actuation_simple)
            
            if not self.vehicle_is_a_bicycle(vehicle_2.type_id):   # Se non è una bici
                if ego_vehicle_wp.is_junction:

                    breaking_distance = 2
                elif self._overtake:
                    breaking_distance = 3
                else:
                    breaking_distance = self._behavior.braking_distance
                if distance_2 < breaking_distance:
                    return self.emergency_stop()
                else:
                    self._local_planner.set_lateral_offset(0)
                    control = self.car_following_manager(vehicle_2, distance_2)
            else: #è una bici
                print("SONO UNA BICI")
                if self.overtake_bike(vehicle_2): #se bici parallela a noi
                    print("OVERTAKE SET")
                    self._local_planner.set_speed(self._speed_limit)
                    print("VELOCITYY: ",get_speed(self._vehicle))
                    control = self._local_planner.run_step(debug=debug) #facciamo il sorpasso
                else:
                    print("OVERTAKE NOT SET")
                    if distance_2 < 7: #se siamo troppo vicini
                        print("\n\n STOP \n\n")
                        return self.emergency_stop() #inchiodiamo
                    else: #altrimenti se non stiamo troppo vicini
                        print("BICI NON PARALLELA,")
                        self._local_planner.set_speed(20) #rallentiamo
                        control = self._local_planner.run_step(debug=debug)

            
            
            
            """
            # Emergency brake if the car is very close.
            #self._behavior.braking_distance
            if distance < 7 and not offset_actuation_simple:
                print("DISTANZA VEICOLO: ",distance)
                print("EMERGENZAAAAAAAAAAA")
                print("VELOCITYY: ",get_speed(self._vehicle))
                return self.emergency_stop()
            elif not self.vehicle_is_a_bicycle(vehicle.type_id):
                print("VEICOLO ELIF :",vehicle)
                print("VELOCITYY: ",get_speed(self._vehicle))
                self._local_planner.set_lateral_offset(0)
                control = self.car_following_manager(vehicle, distance)
            else:
                print("AGGRESSIVEEE")
                self._local_planner.set_speed(self._speed_limit)
                print("VELOCITYY: ",get_speed(self._vehicle))
                control = self._local_planner.run_step(debug=debug)"""

            

        # 3: Intersection behavior

        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]):
            
            print(Back.CYAN+"INCROCIO"+Style.RESET_ALL)

            self._local_planner.set_speed(8)
            
            junction = self._incoming_waypoint.get_junction().id
            print("---- Junction = "+str(junction))
            if self._current_junction != None and self._current_junction == junction:
                print(Back.BLUE+"Incrocio vecchio"+Style.RESET_ALL)
            else:
                print(Back.YELLOW+"Incrocio nuovo"+Style.RESET_ALL)
                self._current_junction = junction
                self._current_straight_yaw = self._world.get_map().get_waypoint(self._vehicle.get_location()).transform.rotation.yaw


            self.set_behavior("cautious")
            
            
            if self.junction_is_free(self._incoming_waypoint.get_junction()): 
                print(Back.LIGHTGREEN_EX+"\n Incrocio libeo \n"+Style.RESET_ALL)
                
                ### Ferdinando

                self._local_planner.set_lateral_offset(0.15)

                ###
                control = self._local_planner.run_step(debug=debug)
            else: 
                print(Back.LIGHTMAGENTA_EX+"\n Incrocio OCCUPATO \n"+Style.RESET_ALL)
                return self.emergency_stop()

        # 4: Normal behavior
        else:
            print(Back.CYAN+"Normal"+Style.RESET_ALL)
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
            print("\nELSE   -----    speed limit = "+ str(self._speed_limit)+'\n')

        return control

