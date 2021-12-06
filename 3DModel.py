#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import stl

import CargoBikeModel as cbm

def stl_write(mesh, file):
    stlmesh = stl.mesh.Mesh(np.zeros(len(mesh.faces), dtype=stl.mesh.Mesh.dtype), name=mesh.options.get('name'))
    for i, f in enumerate(mesh.faces):
        for j in range(3):
            stlmesh.vectors[i][j] = mesh.points[f[j]]
    stlmesh.save(file)

with open('./frame_dict.json', 'r') as dictionary_file:
    frame_params = json.load(dictionary_file)
    
nodes = cbm.get_frame_nodes(frame_params)


# In[2]:


from madcad import *

def divide_bar(R, E):
    return (2*R + E)/4

def generate_rod(v_1, v_2, r):
    length = distance(v_1, v_2)
    base = flatsurface(Circle(Axis(vec3(v_1), normalize(vec3(v_2 - v_1))), r))
    return thicken(base, length, alignment=1)

def generate_tube(V, r, e):
    outline = Circle(Axis(vec3(V[0]), normalize(vec3(V[1] - V[0]))), r)
    path = Wire([vec3(v) for v in V])
    return thicken(tube(outline, path, end=True, section=True), e)

def generate_surface(V):
    outline = wire([Segment(vec3(V[i - 1]), vec3(V[i])) for i in range(len(V))])
    return flatsurface(outline)

L = frame_params['bars_lengths']
R = frame_params['bars_radiuses']
W = frame_params['bars_thicknesses']
N = np.vstack([nodes['N_RF'], nodes['N_FF'][1:]])

dj_spacing = 0.16    # Dropout joints distance
st_height = 0.025    # Seat tube height
ht_height = 0.02     # Handle tube height
ft_length = 0.10     # Fork tube length
ft_radius = 0.019    # Fork tube radius
ft_thickness = 0.003 # Fork tube thickness

dj_width = 0.038     # Dropout joint width
dj_radius = 0.016    # Dropout joint radius
dj_thickness = 0.003 # Dropout joint thickness
bb_width = 0.06      # Bottom bracket width
bb_radius = 0.035    # Bottom bracket radius
bb_thickness = 0.004 # Bottom bracket thickness

vertices = np.array([[N[0, 0], -dj_spacing/2, N[0, 1]],                                # [0]
                     [N[1, 0], 0, N[1, 1]],                                            # [1]
                     [N[2, 0], 0, N[2, 1]],                                            # [2]
                     [N[3, 0], 0, N[3, 1]],                                            # [3]
                     [N[4, 0], 0, N[4, 1]],                                            # [4]
                     [N[5, 0], 0, N[5, 1]],                                            # [5]
                     [N[6, 0], 0, N[6, 1]],                                            # [6]
                     np.insert(N[1] + (N[2] - N[1])*((L[2] + st_height)/L[2]), 1, 0),  # [7]
                     [N[3, 0], 0, N[3, 1] - R[5]],                                     # [8]
                     [N[4, 0], 0, N[4, 1] + ht_height],                                # [9]
                     np.insert(N[6] - (N[7] - N[6])*(R[7]/L[8]), 1, 0),                # [10]
                     np.insert(N[6] + (N[7] - N[6])*((ft_length - R[7])/L[8]), 1, 0)]) # [11]

vertices_y = np.array([[N[0, 0], -(dj_spacing + dj_width)/2, N[0, 1]], # [0]
                       [N[0, 0], -(dj_spacing - dj_width)/2, N[0, 1]], # [1]
                       [N[1, 0], -bb_width/2, N[1, 1]],                # [2]
                       [N[1, 0], bb_width/2, N[1, 1]]])                # [3]

ov_12 = (N[1] - N[0])[::-1]*np.array([1, -1])*(R[0]/L[0])
ov_13 = (N[2] - N[0])[::-1]*np.array([1, -1])*(R[1]/L[1])

vertices_xz = np.insert(np.array([N[0] + ov_12,   # [0]
                                  N[0] - ov_12,   # [1]
                                  N[1] - ov_12,   # [2]
                                  N[1] + ov_12,   # [3]
                                  N[0] + ov_13,   # [4]
                                  N[0] - ov_13,   # [5]
                                  N[2] - ov_13,   # [6]
                                  N[2] + ov_13]), # [7]
                        1, 0, axis=1)


# In[3]:


# BAR 1 - 2
B_12 = generate_tube([vertices[0], vertices[1]], divide_bar(R[0], W[0]), W[0])

H_12 = generate_rod(vertices[0], vertices[1], divide_bar(R[0], W[0]))

XZ_12 = thicken(generate_surface(vertices_xz[0:4]), R[0], alignment=1)

# BAR 0/1 - 3
B_13 = generate_tube([vertices[0], vertices[2]], divide_bar(R[1], W[1]), W[1])

XZ_13 = thicken(generate_surface(vertices_xz[4:8]), R[1], alignment=1)

# BAR 2 - 3 ([8])
B_23 = generate_tube([vertices[1], vertices[7]], R[2], W[2])

H_23 = generate_rod(vertices[1], vertices[7], R[2])

# BAR 2 - 4
B_24 = generate_tube([vertices[1], vertices[3]], R[3], W[3])

# BAR 3 - 5
B_35 = generate_tube([vertices[2], vertices[4]], R[4], W[4])

# BAR 4 ([9]) - 5 ([10])
B_45 = generate_tube([vertices[8], vertices[9]], R[5], W[5])

H_45 = generate_rod(vertices[8], vertices[9], R[5])

# BAR 4 - 6 - 7
B_47 = generate_tube([vertices[3], vertices[5], vertices[6]], R[6], W[6])

# FORK TUBE
B_ft = generate_tube([vertices[10], vertices[11]], ft_radius, ft_thickness)

H_ft = generate_rod(vertices[10], vertices[11], ft_radius)

# Y AXIS TUBES

# DROPOUT JOINTS
B_dj = generate_tube([vertices_y[0], vertices_y[1]], dj_radius, dj_thickness)

H_dj = generate_rod(vertices_y[0], vertices_y[1], dj_radius)

# BOTTOM BRACKET
B_bb = generate_tube([vertices_y[2], vertices_y[3]], bb_radius, bb_thickness)

H_bb = generate_rod(vertices_y[2], vertices_y[3], bb_radius)

print('Bars created!')


# In[4]:


P_12 = difference(difference(difference(B_12, union(H_23, H_bb)), H_dj), XZ_12)

P_13 = difference(difference(difference(B_13, H_23), union(H_12, H_dj)), XZ_13)

P_23 = difference(B_23, H_bb)

P_24 = difference(difference(B_24, H_45), union(H_23, H_bb))

P_35 = difference(difference(B_35, H_45), H_23)

P_45 = B_45

P_47 = difference(difference(B_47, H_45), H_ft)

P_ft = B_ft

P_dj = B_dj

P_bb = B_bb

parts = [P_12, P_13, P_23, P_24, P_35, P_45, P_47, P_ft, P_dj, P_bb]

show(parts)


# In[5]:


for i in range(len(parts)):
    parts[i].finish()
    stl_write(parts[i], './parts/part%d.stl'%(i + 1))
    print('Part %d saved!'%(i + 1))

