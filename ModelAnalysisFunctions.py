#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import sin, cos, sqrt, pi, atan, asin, tan
import numpy as np
import os
import time

import csv
import json

import CargoBikeModel as cbm


# In[ ]:


# GEOMETRY FUNCTIONS

def merge_bars(r, W):
    
    return (4*r - W)/2

def get_trail(lam, r_FW, d_89):
    
    return (r_FW*sin(lam) - d_89)/cos(lam)
    
def get_rear_length(h_RF, r_RW, d_12, d_24):
    
    return sqrt(d_12**2 - (r_RW - h_RF)**2) + d_24
    
def get_front_length(h_RF, lam, d_bw, r_FW, d_78, d_89):
    
    t = asin((r_FW + d_bw)/sqrt(d_78**2 + d_89**2)) - atan(d_89/d_78) - lam
    if t <= 0:
        return d_78*sin(lam) + d_89*cos(lam)
    return (r_FW - h_RF + d_78*cos(lam) - d_89*sin(lam))*tan(t) + d_78*sin(lam) + d_89*cos(lam)


# In[ ]:


# ANALYSIS FUNCTIONS

def get_stable_speeds(bike, speeds=np.linspace(0., 48., num=100)):
    
    evals, evecs = bike.eig(speeds)
    wea, cap, cas = bp.bicycle.sort_modes(evals, evecs)
    stable_range = np.where((np.real(wea['evals'])[:, 0] < 0) &
                            (np.real(wea['evals'])[:, 1] < 0) &
                            (np.real(cap['evals']) < 0) &
                            (np.real(cas['evals']) < 0))[0]
    
    return speeds[stable_range] if stable_range.size else np.array([-1, -1])

def simulate_bikes(system_params, bike_floor_height, cargo_space, steering_angle):

    start_time = time.time()
    
    cargo_bike = cbm.CargoBike('Cargo', system_params)
    
    results = []
    iteration = 0
    total_iterations = len(bike_floor_height)*len(cargo_space)*len(steering_angle)

    r_RW = system_params['frame']['rear_wheel_radius']
    r_FW = system_params['frame']['front_wheel_radius']
    L = system_params['frame']['bars_lengths']
    d_bw = system_params['frame']['d_bw']

    for bfh in bike_floor_height:
        rear_length = get_rear_length(bfh, r_RW, L[0], L[3])
        system_params['frame']['rear_frame_height'] = bfh
        for cs in cargo_space:
            if 'cargo' in system_params:
                system_params['cargo']['cargo_mass'] = system_params['cargo']['mass_function'](cs)
            system_params['frame']['bars_lengths'][6] = cs
            for sa in steering_angle:
                front_length = get_front_length(bfh, sa, d_bw, r_FW, L[8], L[9])
                system_params['frame']['wheelbase'] = rear_length + cs + front_length
                system_params['frame']['steer_axis_tilt'] = sa
                system_params['frame']['trail'] = get_trail(sa, r_FW, L[9])

                cargo_bike.update(system_params)
                stable_speeds = get_stable_speeds(cargo_bike)

                results.append([bfh, cs, sa*180/pi, stable_speeds[0], stable_speeds[-1] - stable_speeds[0]])
                
                iteration += 1
                print('%4d / %d'%(iteration, total_iterations), end = '\r')

    print('%4d / %d'%(iteration, total_iterations))
    print('Completed in %d seconds!'%(time.time() - start_time))
    return results


# In[ ]:


# DATA HANDLING FUNCTIONS

def save_results(results, file_name, file_dir='.'):
    
    table_path = os.path.join(file_dir, file_name + '.csv')

    header = ['bike floor height', 'cargo space', 'steering angle', 'lowest stable speed', 'stable speed range']
    with open(table_path, 'w', encoding='UTF8', newline='') as results_table:
        writer = csv.writer(results_table)
        writer.writerow(header)
        writer.writerows(results)
    print('Table "%s" saved!'%(file_name))
    
def save_parameters(frame_params, file_name, file_dir='.'):
    
    dictionary_path = os.path.join(file_dir, file_name + '.json')

    with open(dictionary_path, 'w') as dictionary_file:
        json.dump(frame_params, dictionary_file)
    print('Dictionary "%s" saved!'%(file_name))

