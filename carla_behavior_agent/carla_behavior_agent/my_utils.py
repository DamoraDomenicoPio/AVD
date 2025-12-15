import carla
import math
import numpy as np

import math
from local_planner import RoadOption, _compute_connection
from misc import get_speed, compute_distance, is_within_distance
import carla


# Assia

def get_forward(yaw):
    yaw_rad = math.radians(yaw)
    forward = carla.Vector3D(
        math.cos(yaw_rad),
        math.sin(yaw_rad),
        0.0
    )
    return forward


def get_angle_between_vectors(vec1, vec2):
    """Returns the angle in degrees between two vectors."""
    v1 = np.array([vec1.x, vec1.y, vec1.z])
    v2 = np.array([vec2.x, vec2.y, vec2.z])
    
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to handle numerical precision errors
    
    return math.degrees(angle)




# Domenico



def calculate_angle_between_waypoints(wp1, wp2):
    dx = wp2.transform.location.x - wp1.transform.location.x
    dy = wp2.transform.location.y - wp1.transform.location.y
    return math.atan2(dy, dx)

# def is_a_curve(map, vehicle, sampling_solution=2, threshold=0.05):
#     ego_vehicle_location = vehicle.get_location()

#     prev_wp = map.get_waypoint(ego_vehicle_location)
#     central_wp = prev_wp.next(sampling_solution)
#     next_wp = central_wp.next(sampling_solution)

#     angle1 = calculate_angle_between_waypoints(prev_wp, central_wp)
#     angle2 = calculate_angle_between_waypoints(central_wp, next_wp)

#     angle_difference = abs(angle2 - angle1)

#     if angle_difference > threshold:
#         print("Curva")
#         return True
#     else:
#         print("Dritto")
#         return False

def get_next_wp(current_waypoint, sampling_solution):
    next_wpt = current_waypoint.next(sampling_solution)[0] 
    return next_wpt

# def is_a_curve(map, vehicle, sampling_solution=2, threshold=20):
#     ego_vehicle_location = vehicle.get_location()

#     prev_wp = map.get_waypoint(ego_vehicle_location)
#     central_wp = get_next_wp(prev_wp, sampling_solution)
#     next_wp = get_next_wp(central_wp, sampling_solution)

#     first_way = _compute_connection(prev_wp, central_wp, threshold)
#     second_way = _compute_connection(central_wp, next_wp, threshold)

#     if first_way != RoadOption.STRAIGHT and second_way != RoadOption.STRAIGHT and first_way == second_way:
#         return True
#     else:
#         return False

def is_a_curve(map, vehicle, sampling_solution=2, threshold=10):
    ego_vehicle_location = vehicle.get_location()

    prev_wp = map.get_waypoint(ego_vehicle_location)
    next_wp = get_next_wp(prev_wp, sampling_solution)

    way = _compute_connection(prev_wp, next_wp, threshold)

    if way != RoadOption.STRAIGHT:
        return True
    else:
        return False
    
def get_vehicle_lane_direction(map, vehicle, opposite=False):
    # Ottenere la trasformazione del veicolo
    transform = vehicle.get_transform()
    vehicle_location = transform.location
    vehicle_rotation = transform.rotation

    # Ottenere il waypoint più vicino al veicolo
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

    # Rotazione del waypoint (direzione della corsia)
    waypoint_rotation = waypoint.transform.rotation

    # Differenza tra la direzione del veicolo e la direzione della corsia
    direction_difference = vehicle_rotation.yaw - waypoint_rotation.yaw

    if opposite:
        direction_difference += 180

    if direction_difference < 0:
        direction_difference += 360

    return direction_difference % 360

    
# def set_speed_vehicle(local_planner, target_speed):
#     local_planner.set_speed(target_speed)


def get_upcoming_waypoints(map, vehicle, distance=8):
    current_waypoint = map.get_waypoint(vehicle.get_location())
    upcoming_waypoints=current_waypoint.next(distance) 
    
    return upcoming_waypoints

def is_waypoint_occupied(world, waypoint, max_distance=4):
    vehicles_list = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles_list:
        distance = waypoint.transform.location.distance(vehicle.get_location())
        if distance <= max_distance:
            return True
    return False

def get_upcoming_waypoints_occupied(world, map, vehicle, distance=8, max_distance=4):
    upcoming_waypoints = get_upcoming_waypoints(map, vehicle, distance)
    
    occupied_waypoints = []
    for waypoint in upcoming_waypoints:
        if is_waypoint_occupied(world, waypoint, max_distance):
            occupied_waypoints.append(waypoint)
    
    return occupied_waypoints


def slow_down(vehicle, local_planner, speed_decrease=5):
    vehicle_speed = get_speed(vehicle)
    target_speed = vehicle_speed - speed_decrease
    local_planner.set_speed(target_speed)
    control = local_planner.run_step()
    return control



def get_vehicle_corners(vehicle):
    # Ottieni la trasformazione del veicolo
    vehicle_transform = vehicle.get_transform()

    # Ottieni l'estensione della bounding box del veicolo
    extent = vehicle.bounding_box.extent

    # Calcola i quattro angoli della bounding box del veicolo
    corners = []

    # Angolo superiore sinistro (frontale)
    corners.append(vehicle_transform.transform(carla.Location(x=extent.x, y=-extent.y)))

    # Angolo superiore destro (frontale)
    corners.append(vehicle_transform.transform(carla.Location(x=extent.x, y=extent.y)))

    # Angolo inferiore destro (posteriore)
    corners.append(vehicle_transform.transform(carla.Location(x=-extent.x, y=extent.y)))

    # Angolo inferiore sinistro (posteriore)
    corners.append(vehicle_transform.transform(carla.Location(x=-extent.x, y=-extent.y)))

    return corners

# def is_car_in_front(my_vehicle, opposing_vehicle):
#     # Ottieni le posizioni dei veicoli
#     my_location = my_vehicle.get_location()
#     opposing_location = opposing_vehicle.get_location()
    
#     # Ottieni le velocità dei veicoli
#     my_velocity = my_vehicle.get_velocity()
#     opposing_velocity = opposing_vehicle.get_velocity()
    
#     # Controlla la direzione del movimento del veicolo opposto
#     relative_velocity = opposing_velocity - my_velocity
#     relative_position = opposing_location - my_location
    
#     # Se il veicolo opposto si sta allontanando, allora ha superato
#     if relative_velocity.x * relative_position.x + relative_velocity.y * relative_position.y > 0:
#         return False
#     else:
#         return True





def is_car_in_front(my_vehicle, target_vehicle, max_distance=50, angle=[0, 90], forward=+1):

    target_transform = target_vehicle.get_transform()
    target_forward_vector = target_transform.get_forward_vector()
    target_extent = target_vehicle.bounding_box.extent.x
    target_rear_transform = target_transform
    target_rear_transform.location = target_rear_transform.location + forward * carla.Location(
        x=target_extent * target_forward_vector.x,
        y=target_extent * target_forward_vector.y,
    )

    return is_within_distance(target_rear_transform, my_vehicle.get_transform(), max_distance, angle)




def check_vehicle_in_front(world, map, vehicle, lane_offset=0, distance=15, check=None, delete_vehicle=[], vehicles=[], angle=[0, 90]):
    '''Check could be: None, forward or backward'''
    if len(vehicles) == 0:
        vehicle_list = world.get_actors().filter("*vehicle*")
    else:
        vehicle_list = vehicles

    ego_vehicle_loc = vehicle.get_location()
    # ego_vehicle_wp = map.get_waypoint(ego_vehicle_loc)
    ego_transform = vehicle.get_transform()
    ego_vehicle_wp = map.get_waypoint(ego_transform.location)


    def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
    vehicle_list = [v for v in vehicle_list if dist(v) < distance and v.id != vehicle.id and v.id not in delete_vehicle]

    # Get the right offset
    if ego_vehicle_wp.lane_id < 0 and lane_offset != 0:
        lane_offset *= -1

    return_vehicle = [False, None, -1]
    vehicle_min_distance = 9999
    for target_vehicle in vehicle_list:

        if check != None:
            if check == "forward":
                if not is_car_in_front(vehicle, target_vehicle, max_distance=distance, angle=angle):
                    continue
            elif check == "backward":
                if is_car_in_front(vehicle, target_vehicle, max_distance=distance, angle=angle):
                    continue

        target_transform = target_vehicle.get_transform()
        target_transform_wp = map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

        target_location = target_vehicle.get_location()
        target_location_wp = map.get_waypoint(target_location)


        # print(ego_vehicle_wp.lane_id, target_transform_wp.lane_id, target_location_wp.lane_id)
        d = 9999  
        if target_transform_wp.road_id == ego_vehicle_wp.road_id and target_transform_wp.lane_id == ego_vehicle_wp.lane_id  + lane_offset:
            # d = compute_distance(target_transform.location, ego_transform.location)
            d = compute_distance(ego_vehicle_loc, target_location)
            if d < vehicle_min_distance:
                vehicle_min_distance = d
                return_vehicle =[True, target_vehicle, d]
        elif target_location_wp.road_id == ego_vehicle_wp.road_id and target_location_wp.lane_id == ego_vehicle_wp.lane_id  + lane_offset:
            # d = compute_distance(target_transform.location, ego_transform.location)
            d = compute_distance(ego_vehicle_loc, target_location)
            if d < vehicle_min_distance:
                vehicle_min_distance = d
                return_vehicle =[True, target_vehicle, d]
    if return_vehicle[0]:
        return return_vehicle[0], return_vehicle[1], return_vehicle[2]
    else:
        return False, None, -1

# def get_vehicle_on_my_lane(world, map, vehicle, lane_offset=0, distance=15):
#     vehicle_list = world.get_actors().filter("*vehicle*")

#     ego_vehicle_loc = vehicle.get_location()
#     ego_vehicle_wp = map.get_waypoint(ego_vehicle_loc)
#     ego_transform = vehicle.get_transform()


#     def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
#     vehicle_list = [v for v in vehicle_list if dist(v) < distance and v.id != vehicle.id]


#     # Get the right offset
#     if ego_vehicle_wp.lane_id < 0 and lane_offset != 0:
#         lane_offset *= -1

#     vehicle_on_my_lane_list = []
#     for target_vehicle in vehicle_list:
#         target_transform = target_vehicle.get_transform()
#         target_vehicle_wp = map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
#         if target_vehicle_wp.road_id == ego_vehicle_wp.road_id:
#             vehicle_on_my_lane_list.append(target_vehicle)
#             corners = get_vehicle_corners(target_vehicle)
#             # Calcola la distanza da ciascun angolo al waypoint della corsia centrale
#             distances_to_center_lane = []
#             min_distance = 9999
#             for corner in corners:
                
#                 # Ottieni il waypoint più vicino sulla stessa corsia
#                 corner_waypoint = world.get_map().get_waypoint(corner, lane_type=carla.LaneType.Driving)

#                 if corner_waypoint.lane_id == ego_vehicle_wp.lane_id:

#                     # Calcola la distanza tra l'angolo e il waypoint della corsia centrale
#                     print()
#                     # if corner_waypoint.lane_offset < min_distance:
#                     #     min_distance = corner_waypoint.lane_offset
#             if min_distance < 9999:
#                 print("MI HA INVASO DI", min_distance)
#                 vehicle_on_my_lane_list.append(target_vehicle)



#             return distances_to_center_lane
#     if len(vehicle_on_my_lane_list) != 0:
#         return True, vehicle_on_my_lane_list
#     else:
#         return False, vehicle_on_my_lane_list


def number_vehicle_in_front(world, map, my_vehicle, distance=15, vehicle_list=[]):
    i = 0

    vehicle_state = True
    last_vehicle = my_vehicle
    backward_vehicles = []
    
    while vehicle_state:
        vehicle_state, other_vehicle, distance_vehicle = check_vehicle_in_front(world, map, last_vehicle, distance=distance, vehicles=vehicle_list, check="forward", delete_vehicle=backward_vehicles)
        print("LOOP", vehicle_state, other_vehicle, distance_vehicle)
        backward_vehicles.append(last_vehicle.id)
        # print(my_vehicle.id, backward_vehicles)
        if vehicle_state:
            i += 1
            last_vehicle = other_vehicle
        
    
    if my_vehicle == last_vehicle:
        return i, -1, last_vehicle

    else:
        return i, compute_distance(my_vehicle.get_location(), last_vehicle.get_location()), last_vehicle


def check_overtake(world, map, my_vehicle, max_distance=30, distance_tail=15, vehicles=[], sampling_resolution=2):
    # TODO: aggiungi controllo sull'incrocio
    if map.get_waypoint(my_vehicle.get_location()).is_junction:
        return False
    vehicle_state, vehicle, distance_before_overtake = check_vehicle_in_front(world, map, my_vehicle, lane_offset=0, distance=max_distance, vehicles=vehicles, check="forward")
    # if get_speed(my_vehicle) < 0.1:
        
    #     if vehicle_state:
    #         # Ritorna False se sto ad un incrocio o l'ultima macchina della fila sta in un incorcio
    #         num, distance, last_vehicle = number_vehicle_in_front(world, map, vehicle, distance=distance_tail)
    #         return not map.get_waypoint(last_vehicle.get_location()).is_junction
        
    if vehicle_state:
        if get_speed(vehicle) < 0.1 and get_speed(my_vehicle) < 0.1:  # Se la macchina è ferma e l'ego vehicle è fermo ...
            # Ritorna False se l'ultima macchina sta in un incrocio
            num, distance, last_vehicle = number_vehicle_in_front(world, map, vehicle, distance=distance_tail)
            print("ULTIMA MACCHINA:", num, distance, last_vehicle, map.get_waypoint(last_vehicle.get_location()).next(5)[0].is_junction)
            
            w = map.get_waypoint(last_vehicle.get_location())
            for i in range(10):
                if w.next(sampling_resolution)[0].is_junction:
                    return False
                w = w.next(sampling_resolution)[0]
            return True
            
    return False


def wait_overtaking(world, map, my_vehicle,max_distance=30, distance_tail=15, begin_distance=10, vehicles=[]):
    vehicle_state, first_vehicle, distance_from_first_vehicle = check_vehicle_in_front(world, map, my_vehicle, lane_offset=0, distance=max_distance, vehicles=vehicles, check="forward")
    num, distance_from_last_vehicle, last_vehicle = number_vehicle_in_front(world, map, first_vehicle, distance=distance_tail, vehicle_list=vehicles)
    d = compute_distance(my_vehicle.get_location(), last_vehicle.get_location())
    if distance_from_first_vehicle <= begin_distance:
        return False, d
    else:
        return True, -1



def check_to_right(world, map, my_vehicle, distance=15, vehicles=[]):
    check, _, _ = check_vehicle_in_front(world, map, my_vehicle, lane_offset=-2, distance=distance, vehicles=vehicles, check="forward", angle = [0, 90])
    return not check
        