#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import sin, cos, sqrt, pi, atan2, dist # For the math
from uncertainties import ufloat, ufloat_fromstr # For the values with uncertainty
import bicycleparameters as bp                   # For the bicycle equations
import matplotlib.pyplot as plt                  # For the graphs graphs
import os                                        # For the external files
import traceback                                 # For the error handling
import numpy as np                               # For the arrays


# In[ ]:


# GEOMETRY

def find_vertex(p_1, p_2, d_13, d_23):
    # Returns the third vertex of a triangle, given the first vertex p_1,
    # the second vertex p_2, the distance d_13 between p_1 and the third vertex,
    # and the distance d_23 between p_2 and the third vertex
    
    x_1, z_1 = p_1
    x_2, z_2 = p_2
    
    d_12 = dist(p_1, p_2)
    
    c_1 = (d_12**2 + d_13**2 - d_23**2)/(2*d_12)
    c_2 = sqrt(d_13**2 - c_1**2)
    
    return [x_1 + (c_1*(x_2 - x_1) + c_2*(z_1 - z_2))/d_12,
            z_1 + (c_1*(z_2 - z_1) + c_2*(x_2 - x_1))/d_12]

def rotate_points(P, t):
    # Rotates the points P by t radians, relative to the origin
    
    R = np.array([[cos(t), sin(t)], [-sin(t), cos(t)]])
    return np.dot(P, R)

def angle(a, b):
    # Returns the slope of the ab line
    
    return atan2(*(b - a)[::-1])

def midpoint(a, b):
    # Returns the midpoint of the ab line
    
    return np.mean([a, b], axis=0)


# In[ ]:


# INERTIA

def disk_inertia(m, r):
    # Returns the inertia tensor of a disk of radius r and mass m
    
    Ixx = m*r**2/4
    Iyy = 2*Ixx
    
    I = np.array([[Ixx,   0,   0],
                  [  0, Iyy,   0],
                  [  0,   0, Ixx]])
    
    return I

def bar_inertia(m, a, b):
    # Returns the inertia tensor of bar of mass m, defined by the 
    # points a and b, with respect to its midpoint
    
    l = dist(a, b)  # Bar length
    t = angle(a, b) # Bar angle
    
    Ixx = m*(l*sin(t))**2/12
    Izz = m*(l*cos(t))**2/12
    Iyy = Ixx + Izz
    Ixz = -m*l**2*sin(t)*cos(t)/12
    
    I = np.array([[Ixx,   0, Ixz],
                  [  0, Iyy,   0],
                  [Ixz,   0, Izz]])
    
    return I

def parallel_axis_theorem(I, m, h):
    # Applies the parallel axis theorem to a body with inertia tensor I
    # and mass m, across the vector h
    # Returns the new inertia tensor
    
    x, z = h
    
    Ixx_h = m*z**2
    Izz_h = m*x**2
    Iyy_h = Ixx_h + Izz_h
    Ixz_h = -m*x*z
    
    I_o = I + np.array([[Ixx_h,     0, Ixz_h],
                        [    0, Iyy_h,     0],
                        [Ixz_h,     0, Izz_h]])
    
    return I_o

def offset_bar_inertia(m, a, b, o):
    # Returns the inertia tensor of an offset bar of mass m, defined
    # by the points a and b, with respect to the point o
    
    c = midpoint(a, b) # Bar midpoint
    h = o - c          # Offset vector
    
    I = parallel_axis_theorem(bar_inertia(m, a, b), m, h)
    
    return I

def frame_inertia(M, B, g):
    # Returns the inertia tensor of a frame, comprised of bars B
    # of masses M, with respect to the point g
    
    I = np.sum([offset_bar_inertia(m, *b, g)
                for m, b in zip(M, B)], axis=0)
    
    return I


# In[3]:


# NODES

def get_frame_nodes(frame_params):
    # Receives a dictionary containing the parameters of the bike frame
    # Returns a dictionary containing the nodes representing the frame's vertices
    
    # Retrieves the necessary parameters from the dictionary
    r_RW = frame_params['rear_wheel_radius']
    r_FW = frame_params['front_wheel_radius']
    h_RF = frame_params['rear_frame_height']
    lam = frame_params['steer_axis_tilt']
    w = frame_params['wheelbase']
    c = frame_params['trail']
    L = frame_params['bars_lengths']

    # Rear frame nodes --------------------------------------------------------------
    N_RF = np.zeros([7, 2]) # Init. of the nodes (x, z) for the RF
    
    N_RF[0] = [0, r_RW]                                 # (1) - Rear wheel node
    
    N_RF[1] = [sqrt(L[0]**2 - (r_RW - h_RF)**2), h_RF]  # (2)
    N_RF[2] = find_vertex(N_RF[0], N_RF[1], L[1], L[2]) # (3)
    N_RF[3] = N_RF[1] + [L[3], 0]                       # (4)
    N_RF[4] = N_RF[3] + [0, L[5]]                       # (5)
    N_RF[5] = N_RF[3] + [L[6], 0]                       # (6)
    
    L[4] = dist(N_RF[2], N_RF[4])                       # 3-5 bar length
    
    # Front frame nodes -------------------------------------------------------------
    N_FF = np.zeros([3, 2]) # Init. of the nodes (x, z) for the FF
    
    N_FF[2] = [w, r_FW]                                 # (9) - Front wheel node
    
    L[9] = r_FW*sin(lam) - c*cos(lam)                   # 8-9 bar length

    N_FF[1] = N_FF[2] - [L[9]*cos(lam), L[9]*sin(lam)]  # (8)
    N_FF[0] = N_FF[1] - [L[8]*sin(lam), -L[8]*cos(lam)] # (7)
    N_RF[6] = N_FF[0]

    L[7] = dist(N_RF[5], N_FF[0])                       # 6-7 bar length
    
    # Carvallo-Whipple Model representation nodes ----------------------------------
    N_m = np.zeros([4, 2]) # Init. of the nodes (x, z) for the c.w. model
    
    d_17 = dist(N_RF[0], N_FF[0])
    d_18 = dist(N_RF[0], N_FF[1])
    d_e7 = (d_18**2 - d_17**2 - L[8]**2)/(2*L[8])
    
    N_m[0] = N_RF[0]
    N_m[1] = find_vertex(N_RF[0], N_FF[1], sqrt(d_17**2 - d_e7**2), L[8] + d_e7)
    N_m[2] = N_FF[1]
    N_m[3] = N_FF[2]
    
    # Nodes dictionary
    nodes = {
        'N_RF': N_RF,
        'N_FF': N_FF,
        'N_m':  N_m
    }
    
    return nodes

def add_rider_nodes(nodes, rider_params):
    # Receives the nodes dictionary and a dictionary containing the parameters of
    # the bike's rider
    # Adds the rider's nodes to the nodes dictionary
    
    # Retrieves the necessary parameters from the dictionaries
    h_Rs = rider_params['saddle_height']
    h_Rh = rider_params['handlebar_height']
    a_R = rider_params['lean_angle']
    L_R = rider_params['limbs_lengths']
    N_RF = nodes['N_RF']
    
    # Rider nodes -------------------------------------------------------------------
    N_R = np.zeros([7, 2]) # Init. of the nodes (x, z) for the rider
    
    d_21 = dist(N_RF[2], N_RF[1])
    
    N_R[0] = N_RF[1]                                     # (1)
    N_R[2] = N_RF[2] + (N_RF[1] - N_RF[2])*(-h_Rs/d_21)  # (3)
    N_R[1] = find_vertex(N_R[2], N_R[0], L_R[0], L_R[1]) # (2)
    N_R[3] = N_R[2] + [d_21*cos(a_R), d_21*sin(a_R)]     # (4)
    N_R[5] = N_RF[4] + [0, h_Rh]                         # (6)
    N_R[4] = find_vertex(N_R[5], N_R[3], L_R[4], L_R[3]) # (5)
    N_R[6] = N_R[3] + [L_R[5]*cos(a_R), L_R[5]*sin(a_R)] # (7)
    
    # Adds the nodes to the dictionary
    nodes['N_R'] = N_R

def add_cargo_nodes(nodes, cargo_params):
    # Receives the nodes dictionary and a dictionary containing the parameters of
    # the bike's cargo
    # Adds the cargo's nodes to the nodes dictionary
    
    # Retrieves the necessary parameters from the dictionaries
    h_C = cargo_params['cargo_height']
    N_RF = nodes['N_RF']
    
    # Cargo nodes -------------------------------------------------------------------
    N_C = np.zeros([4, 2]) # Init. of the nodes (x, z) for the cargo
    
    N_C[0] = N_RF[3]            # (1)
    N_C[1] = N_RF[3] + [0, h_C] # (2)
    N_C[2] = N_RF[5]            # (3)
    N_C[3] = N_RF[5] + [0, h_C] # (4)
    
    # Adds the nodes to the dictionary
    nodes['N_C'] = N_C
    
def get_nodes(system_params):
    # Receives a dictionary containing the parameters of all the bodies in the
    # system
    # Returns a dictionary containing the nodes of these bodies
    
    # Creates the dictionary, initially containing the frame nodes
    nodes = get_frame_nodes(system_params['frame'])
    
    # Adds the rider nodes to the dictionary, if the system has a rider
    if 'rider' in system_params:
        add_rider_nodes(nodes, system_params['rider'])
    
    # Adds the cargo nodes to the dictionary, if the system has cargo
    if 'cargo' in system_params:
        add_cargo_nodes(nodes, system_params['cargo'])
    
    return nodes

def invert_z_axis(nodes):
    # Receives the nodes dictionary
    # Inverts the z values of all the nodes
    
    nodes['N_RF'][:, 1] *= -1
    nodes['N_FF'][:, 1] *= -1
    
    if 'N_R' in nodes:
        nodes['N_R'][:, 1] *= -1
        
    if 'N_C' in nodes:
        nodes['N_C'][:, 1] *= -1
    
    nodes['N_m'][:, 1] *= -1
    
def change_basis(nodes):
    # Receives the nodes dictionary
    # Moves each set of nodes to their respective basis, as described by the model
    
    n_p = nodes['N_m'][1]          # Model's pivot between the rear and front frames
    a_p = angle(*nodes['N_m'][:2]) # Angle between the pivot node and the X-axis

    # Rotates and displaces the nodes to their appropriate sub-spaces
    
    nodes['N_RF'] = rotate_points(nodes['N_RF'] - n_p, -a_p)
    nodes['N_FF'] = rotate_points(nodes['N_FF'] - n_p, -a_p)
    
    if 'N_R' in nodes:
        nodes['N_R'] = rotate_points(nodes['N_R'] - n_p, -a_p)
    
    if 'N_C' in nodes:
        nodes['N_C'] = rotate_points(nodes['N_C'] - n_p, -a_p)


# In[5]:


# MECHANICS

def get_bars(nodes):
    # Receives the nodes dictionary
    # Returns a dictionary containing the connections between the nodes
    
    # Retrieves the necessary parameters from the dictionary
    N_RF = nodes['N_RF']
    N_FF = nodes['N_FF']
    N_m = nodes['N_m']
    
    # Rear frame bars ---------------------------------------------------------------
    B_RF = np.zeros([8, 2, 2]) # Init. of the bars [(xi, zi), (xj, zj)] for the RF
    
    B_RF[0] = [N_RF[0], N_RF[1]] # 1-2 bar connection
    B_RF[1] = [N_RF[0], N_RF[2]] # 1-3 bar connection
    B_RF[2] = [N_RF[1], N_RF[2]] # 2-3 bar connection
    B_RF[3] = [N_RF[1], N_RF[3]] # 2-4 bar connection
    B_RF[4] = [N_RF[2], N_RF[4]] # 3-5 bar connection
    B_RF[5] = [N_RF[3], N_RF[4]] # 4-5 bar connection
    B_RF[6] = [N_RF[3], N_RF[5]] # 4-6 bar connection
    B_RF[7] = [N_RF[5], N_RF[6]] # 6-7 bar connection
    
    # Front frame bars --------------------------------------------------------------
    B_FF = np.zeros([2, 2, 2]) # Init. of the bars [(xi, zi), (xj, zj)] for the FF
    
    B_FF[0] = [N_FF[0], N_FF[1]] # 7-8 bar connection
    B_FF[1] = [N_FF[1], N_FF[2]] # 8-9 bar connection
    
    # Carvallo-Whipple Model representation bars ------------------------------------
    B_m = np.zeros([3, 2, 2])
    B_m[0] = [N_m[0], N_m[1]]
    B_m[1] = [N_m[1], N_m[2]]
    B_m[2] = [N_m[2], N_m[3]]
    
    # Bars dictionary
    bars = {
        'B_RF': B_RF,
        'B_FF': B_FF,
        'B_m':  B_m
    }
    
    # Rider bars --------------------------------------------------------------------
    if 'N_R' in nodes:
        N_R = nodes['N_R']
        B_R = np.zeros([6, 2, 2]) # Init. of the bars [(xi, zi), (xj, zj)] for the R
        B_R[0] = [N_R[0], N_R[1]] # 1-2 bar connection
        B_R[1] = [N_R[1], N_R[2]] # 2-3 bar connection
        B_R[2] = [N_R[2], N_R[3]] # 3-4 bar connection
        B_R[3] = [N_R[3], N_R[4]] # 4-5 bar connection
        B_R[4] = [N_R[4], N_R[5]] # 5-6 bar connection
        B_R[5] = [N_R[3], N_R[6]] # 4-7 bar connection
        bars['B_R'] = B_R
    
    # Cargo bars --------------------------------------------------------------------
    if 'N_C' in nodes:
        N_C = nodes['N_C']
        B_C = np.zeros([2, 2, 2]) # Init. of the bars [(xi, zi), (xj, zj)] for the C
        B_C[0] = [N_C[0], N_C[3]] # 1-4 bar connection
        B_C[1] = [N_C[1], N_C[2]] # 2-3 bar connection
        bars['B_C'] = B_C
    
    return bars

def get_masses(system_params):
    # Receives a dictionary containing the parameters of all the bodies in the
    # system
    # Returns a dictionary containing the masses of each connection of these bodies
    # nodes
    
    # Frame masses ------------------------------------------------------------------
    # M = A(R, E)*L*p = (pi*E*(2*R - E))*L*p
    p = system_params['frame']['material_density']
    M_F = [(pi*w*(2*r - w))*l*p 
           for w, r, l
           in zip(system_params['frame']['bars_thicknesses'],
                  system_params['frame']['bars_radiuses'],
                  system_params['frame']['bars_lengths'])]
    
    # Masses dictionary
    masses = {
        'M_W':  np.array([system_params['frame']['rear_wheel_mass'],
                          system_params['frame']['front_wheel_mass']]),
        'M_RF': np.array(M_F[0:8]),
        'M_FF': np.array(M_F[8:10])
    }
    
    # Rider masses ------------------------------------------------------------------
    if 'rider' in system_params:
        masses['M_R'] = np.array(system_params['rider']['limbs_masses'])
    
    # Cargo masses ------------------------------------------------------------------
    if 'cargo' in system_params:
        masses['M_C'] = np.array([system_params['cargo']['cargo_mass']/2]*2)
    
    return masses

def get_mechanical_properties(masses, bars):
    # Receives the masses and bars dictionaries
    # Returns a dictionary containing the mechanical properties of each body in the
    # model
    
    # Retrieves and groups the masses and bars of the rear frame body, which may
    # include the rider and the cargo
    M_RF = masses['M_RF']
    B_RF = bars['B_RF']
    if 'M_R' in masses and 'B_R' in bars:
        M_RF = np.hstack([M_RF, masses['M_R']])
        B_RF = np.vstack([B_RF, bars['B_R']])
    if 'M_C' in masses and 'B_C' in bars:
        M_RF = np.hstack([M_RF, masses['M_C']])
        B_RF = np.vstack([B_RF, bars['B_C']])
    
    # Retrieves the masses and bars of the front frame
    M_FF = masses['M_FF']
    B_FF = bars['B_FF']
    
    # Retrieves the bars representing the Carvallo-Whipple Model
    B_m = bars['B_m']
    
    n_p = B_m[0, 1]      # Model's pivot between the rear and front frames
    a_p = angle(*B_m[0]) # Angle between the pivot node and the X-axis
    
    r_RW = B_m[0, 0, 1]  # Rear wheel radius
    r_FW = B_m[2, 1, 1]  # Front wheel radius
    
    # Retrieves the masses of the wheels
    m_RW, m_FW = masses['M_W']
    
    # Local centers of mass --------------------------------------------------------
    g_RF = np.average(np.mean(B_RF, axis=1), axis=0, weights=M_RF)
    g_FF = np.average(np.mean(B_FF, axis=1), axis=0, weights=M_FF)
    
    # Global centers of mass -------------------------------------------------------
    g_RFm = rotate_points([g_RF], a_p)[0] + n_p
    g_FFm = rotate_points([g_FF], a_p)[0] + n_p

    # Mechanical properties dictionary
    mechanical_properties = {
        'm_RF':  np.sum([M_RF]),                  # Total mass of the RF
        'm_FF':  np.sum([M_FF]),                  # Total mass of the FF
        'g_RFm': g_RFm,                           # Global center of mass of the RF
        'g_FFm': g_FFm,                           # Global center of mass of the FF
        'I_RF':  frame_inertia(M_RF, B_RF, g_RF), # Inertia tensor of the RF
        'I_FF':  frame_inertia(M_FF, B_FF, g_FF), # Inertia tensor of the FF
        'I_RW':  disk_inertia(m_RW, r_RW),        # Inertia tensor of the RW
        'I_FW':  disk_inertia(m_FW, r_FW)         # Inertia tensor of the FW
    }
    
    return mechanical_properties


# In[41]:


# BENCHMARK

def create_benchmark_dictionary(system_params, mechanical_properties):
    # Receives a dictionary containing the parameters of all the bodies in the
    # system and a dictionary containing their mechanical properties
    # Returns the "Benchmark" dictionary
    
    r_RW = system_params['frame']['rear_wheel_radius']
    r_FW = system_params['frame']['front_wheel_radius']
    w = system_params['frame']['wheelbase']
    lam = system_params['frame']['steer_axis_tilt']
    c = system_params['frame']['trail']
    m_RW = system_params['frame']['rear_wheel_mass']
    m_FW = system_params['frame']['front_wheel_mass']
    
    m_RF = mechanical_properties['m_RF']
    m_FF = mechanical_properties['m_FF']
    g_RFm = mechanical_properties['g_RFm']
    g_FFm = mechanical_properties['g_FFm']
    I_RF = mechanical_properties['I_RF']
    I_FF = mechanical_properties['I_FF']
    I_RW = mechanical_properties['I_RW']
    I_FW = mechanical_properties['I_FW']
    
    benchmark_dictionary = {
        'w':    ufloat(w, 0),
        'c':    ufloat(c, 0),
        'lam':  ufloat(lam, 0),
        'g':    ufloat(9.81, 0),
        'rR':   ufloat(r_RW, 0),
        'mR':   ufloat(m_RW, 0),
        'IRxx': ufloat(I_RW[0, 0], 0),
        'IRyy': ufloat(I_RW[1, 1], 0),
        'xB':   ufloat(g_RFm[0], 0),
        'zB':   ufloat(g_RFm[1], 0),
        'mB':   ufloat(m_RF, 0),
        'IBxx': ufloat(I_RF[0, 0], 0),
        'IByy': ufloat(I_RF[1, 1], 0),
        'IBzz': ufloat(I_RF[2, 2], 0),
        'IBxz': ufloat(I_RF[0, 2], 0),
        'xH':   ufloat(g_FFm[0], 0),
        'zH':   ufloat(g_FFm[1], 0),
        'mH':   ufloat(m_FF, 0),
        'IHxx': ufloat(I_FF[0, 0], 0),
        'IHyy': ufloat(I_FF[1, 1], 0),
        'IHzz': ufloat(I_FF[2, 2], 0),
        'IHxz': ufloat(I_FF[0, 2], 0),
        'rF':   ufloat(r_FW, 0),
        'mF':   ufloat(m_FW, 0),
        'IFxx': ufloat(I_FW[0, 0], 0),
        'IFyy': ufloat(I_FW[1, 1], 0)
    }
    
    return benchmark_dictionary

def save_benchmark_file(bike_name, file_dir, benchmark_dictionary):
    # Saves the benchmark dictionary to a text file, in the file_dir directory
    # Returns the string sent to the file

    file_path = os.path.join(file_dir, bike_name + 'Benchmark.txt')
    
    # Formato do .txt para armazenamento dos dados
    if type(benchmark_dictionary) is dict:
        parameters_string = '\n'.join('{:s} = {:.6f}'.format(key, value)
                                      for key, value
                                      in benchmark_dictionary.items())
        mode = 'w'
    else:
        parameters_string = ''
        mode = 'a'
    
    os.makedirs(file_dir, exist_ok=True)
    with open(file_path, mode) as parameters_file:
        parameters_file.write(parameters_string)
        
    return parameters_string

def read_benchmark_file(file_path):
    # Reads the benchmark text file in the file_path and converts it to a
    # dictionary
    # Returns the benchmark dictionary
    
    # Based on bicycleparameters.io
    benchmark_dictionary = {}
    with open(file_path, 'r') as parameters_file:
        for line in parameters_file:
            # Ignores lines that start with a hash
            if line[0] != '#':
                # Removes any whitespace characters and comments at the end of
                # the line, then split the right and left side of the equality
                equality = line.strip().split('#')[0].split('=')
                # ['a ', ' 0.1 +/- 0.05']
                if '+/-' in equality[1]:
                    value = ufloat_fromstr(equality[1])
                else:
                    value = float(equality[1])
                # Stores line in the dictionary
                benchmark_dictionary[equality[0].strip()] = value
    
    return benchmark_dictionary

def check_benchmark_dictionary(benchmark_dictionary):
    # Checks if the benchmark dictionary has all the parameters needed in it
    # Returns True if it has all of its necessary parameters
    # Returns False if it's missing a parameter, and prints out its key
    
    key_list = ['w', 'c', 'lam', 'g',
                'rR', 'mR', 'IRxx', 'IRyy',
                'xB', 'zB', 'mB', 'IBxx', 'IByy', 'IBzz', 'IBxz',
                'xH', 'zH', 'mH', 'IHxx', 'IHyy', 'IHzz', 'IHxz',
                'rF', 'mF', 'IFxx', 'IFyy']
    
    for key in key_list:
        if key not in benchmark_dictionary:
            print("{0} is missing!".format(key))
            return False
    return True


# In[90]:


# PLOT

def draw_circle(c, r, color):
    # Plots a circle of center c and radius r

    t = np.linspace(0, 2*pi, 50)
    x = r*np.cos(t) + c[0]
    z = r*np.sin(t) + c[1]
    plt.plot(x, z, color=color)

def draw_bar(b, color):
    # Plots the bar b
    
    x, z = np.transpose(b)
    plt.plot(x, z, color=color)

def draw_geometry(system_params, figure=None, show=True):
    # Receives a dictionary containing the parameters of all the bodies in the
    # system
    # Plots the geometry of the bodies in the system and returns the figure
    
    nodes = get_nodes(system_params)
    bars = get_bars(nodes)
    
    # If a figure is not given as an argument, it generates its own
    if not figure:
        figure = plt.figure(figsize=(16, 8))
        plt.title('Cargo Bike Geometry')
        plt.xlabel('x [m]')
        plt.ylabel('-z [m]')
        plt.axis('equal')
    
    B_m = bars['B_m']
    draw_circle(B_m[0, 0], B_m[0, 0, 1], 'grey') # Plots the rear wheel
    draw_circle(B_m[2, 1], B_m[2, 1, 1], 'grey') # Plots the front wheel
    
    # Plots the rear frame
    for b_RF in bars['B_RF']:
        draw_bar(b_RF, 'blue')
    
    # Plots the front frame
    for b_FF in bars['B_FF']:
        draw_bar(b_FF, 'green')
    
    # Plots a rider if there's one
    if 'B_R' in bars:
        for b_R in bars['B_R']:
            draw_bar(b_R, 'grey')
    
    # Plots the cargo if there's some
    if 'B_C' in bars:
        for b_C in bars['B_C']:
            draw_bar(b_C, 'grey')
    
    if show:
        plt.show()
    
    return figure


# In[104]:


# CARGOBIKE

def get_benchmark_params(system_params):
    # Receives a dictionary containing the parameters of all the bodies in the
    # system
    # Returns the "Benchmark" dictionary
    
    nodes = get_nodes(system_params)
    invert_z_axis(nodes)
    change_basis(nodes)
    bars = get_bars(nodes)
    masses = get_masses(system_params)
    mechanical_properties = get_mechanical_properties(masses, bars) 
    
    return create_benchmark_dictionary(system_params, mechanical_properties)

class CargoBike(object):
    
    def __new__(cls, bike_name, system_params=None, work_dir='.'):
        # Creates a new instance to check the arguments passed
        # If everything is in order, returns an object
        
        cargo_bike = super(CargoBike, cls).__new__(cls)
        
        cargo_bike.file_dir = os.path.join(work_dir, 'bicycles', bike_name, 'Parameters')
        file_path = os.path.join(cargo_bike.file_dir, bike_name + 'Benchmark.txt')
        
        cargo_bike.parameters = {'Benchmark': {}}
        
        if system_params:
            print("Let me get some benchmark parameters out of these arguments...")
            # If some parameters were passed as an argument, calculates the benchmark parameters
            try:
                benchmark_dictionary = get_benchmark_params(system_params)
                cargo_bike.parameters['Benchmark'] |= benchmark_dictionary
                cargo_bike.parameters |= system_params
            except:
                print("Could you double-check them for me?\n"
                      "I couldn't get past this...\n\n{0}"
                      .format(traceback.format_exc()))
                # If it can't compute the parameters, no object is created
                return None
            
        elif os.path.isfile(file_path):
            print("I've found a benchmark file at {0}!\n".format(cargo_bike.file_dir))
            # If there is a benchmark parameters file, retrieves its content
            try:
                benchmark_dictionary = read_benchmark_file(file_path)
                if check_benchmark_dictionary(benchmark_dictionary):
                    cargo_bike.parameters['Benchmark'] |= benchmark_dictionary
                else:
                    print("That's not gonna work!")
                    # If the file is missing parameters, no object is created
                    return None
            except:
                print("Could you take a look at it for me?\n"
                      "I couldn't get past this...\n\n{0}"
                      .format(traceback.format_exc()))
                # If there's something wrong with the file, no object is created
                return None
        else:
            print("I don't have any data to work with!\n"
                  "Make sure to pass me some arguments or have a benchmark file ready at {0}!\n"
                  .format(cargo_bike.file_dir))
            # If there's no data available to retrieve, no object is created
            return None
        
        print("Alright! Everything seems to be in order!\n")
        return cargo_bike
        
    def __init__(self, bike_name, system_params=None, work_dir='.'):
        
        self.bicycleName = bike_name
        
        if 'rider' in self.parameters:
            self.hasRider = True
            self.riderName = system_params['rider']['rider_name']
        else:
            self.hasRider = False
        
    def update(self, system_params=None):
        # Updates the object with either new arguments or a new file in its directory
        
        file_path = os.path.join(self.file_dir, self.bicycleName + 'Benchmark.txt')
        
        if system_params:
            # If some parameters were passed as an argument, calculates the benchmark parameters
            try:
                benchmark_dictionary = get_benchmark_params(system_params)
                self.parameters['Benchmark'] |= benchmark_dictionary
                self.parameters |= system_params
                
                if 'rider' in self.parameters:
                    self.hasRider = True
                    self.riderName = system_params['rider']['rider_name']
                else:
                    self.hasRider = False
                    
            except:
                print("Could you check these arguments for me?\n"
                      "I couldn't get past this...\n\n{0}"
                      .format(traceback.format_exc()))
                # If it can't compute the parameters, the object isn't updated
                
        elif os.path.isfile(file_path):
            # If there is a benchmark parameters file, retrieves its content
            try:
                benchmark_dictionary = read_benchmark_file(file_path)
                if check_benchmark_dictionary(benchmark_dictionary):
                    self.parameters['Benchmark'] |= benchmark_dictionary
                else:
                    print("That's not gonna work!")
                    # If the file is missing parameters, the object isn't updated
                
            except:
                print("Could you check the benchmark file for me?\n"
                      "I couldn't get past this...\n\n{0}"
                      .format(traceback.format_exc()))
                # If there's something wrong with the file, the object isn't updated
        
        else:
            print("I don't have any data to update!\n"
                  "Make sure to pass me some arguments or have a benchmark file ready at {0}!"
                  .format(cargo_bike.file_dir))
            # If there's no data available to retrieve, the object isn't updated
        
    def save_benchmark(self):
        # Saves the benchmark file in its parameters to a text file in its directory
        
        save_benchmark_file(self.bicycleName, self.file_dir, self.parameters['Benchmark'])
        print("Benchmark parameters saved at {0}\n".format(self.file_dir))
        
    def show(self):
        # Plots the geometry of the system
        
        if 'frame' in self.parameters:
            figure = plt.figure(figsize=(16, 8))
            plt.title('{0} Geometry'.format(self.bicycleName))
            plt.xlabel('x [m]')
            plt.ylabel('-z [m]')
            plt.axis('equal')
            return draw_geometry(self.parameters, figure)
        else:
            print("Can't show you anything yet...\n"
                  "Try passing me some arguments first!\n")
    
    # Methods inherited from the Bicycle class from bicycleparameters!
    # Documentation available at https://pythonhosted.org/BicycleParameters/index.html
    # (13/09/2021)
    __str__ = bp.Bicycle.__str__
    canonical = bp.Bicycle.canonical
    state_space = bp.Bicycle.state_space
    eig = bp.Bicycle.eig
    plot_eigenvalues_vs_speed = bp.Bicycle.plot_eigenvalues_vs_speed
    plot_bode = bp.Bicycle.plot_bode
    compare_bode_speeds = bp.Bicycle.compare_bode_speeds

