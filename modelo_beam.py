from numpy import array, pi, zeros, ix_
import numpy as np
from beam_element import cst_planestress, cst_planestress_post
from scipy.linalg import solve

fid = open("/Users/Fran/Desktop/Gmesh/beam5m3.msh","r")

LINE_ELEMENT = 1
TRI_ELEMENT = 2
Fijo = 1
Viga = 2

while True :
    line = fid.readline()
    
    if line.find("$Nodes") >= 0 :
        break

Nnodes = int(fid.readline())

#print(f"Nnodes = {Nnodes}")

xy = np.zeros([Nnodes, 2])

for i in range(Nnodes) :
    line = fid.readline()
    sl = line.split()
    xy[i, 0] = float(sl[1])
    xy[i, 1] = float(sl[2])

print(f"xy = {xy}")

print(f"Nnodes = {Nnodes}")

while True :
    line = fid.readline()
    
    if line.find("$Elements") >= 0 :
        break   

Nelements = int(fid.readline())

print(f"Nelements = {Nelements}")

conec = np.zeros((Nelements,3), dtype = np.int32)

fixed_nodes = []

Ntriangles = 0
Triangles = []
for i in range(Nelements) :
    line = fid.readline()
    sl = line.split()
    element_number = np.int32(sl[0]) - 1
    element_type = np.int32(sl[1])
    physical_grp = np.int32(sl[3])
    entity_number = np.int32(sl[4])
    
    if element_type == LINE_ELEMENT and physical_grp == Fijo: # linea tipo fija
        n1 = np.int32(sl[5]) - 1
        n2 = np.int32(sl[6]) - 1
        fixed_nodes += [n1, n2]
        
    if element_type == TRI_ELEMENT and physical_grp == Viga: # linea tipo viga
        n0 = np.int32(sl[5]) - 1
        n1 = np.int32(sl[6]) - 1
        n2 = np.int32(sl[7]) - 1
        
        conec[element_number, :] = [n0, n1, n2]
        
        Triangles.append(element_number)
        Ntriangles += 1

fid.close()
   
print(conec)

NDOFs = 2*Nnodes

properties = {}

rho = 2500
g = 9.81

properties["E"] = 20e9
properties["nu"] = 0.25
properties["bx"] = 0
properties["by"] = -rho*g
properties["t"] = 0.5

K = zeros((NDOFs, NDOFs))
f = zeros((NDOFs, 1))

for e in Triangles:
    ni = conec[e,0]
    nj = conec[e,1]
    nk = conec[e,2]

    print(f"e = {e} ni = {ni} nj = {nj} nk = {nk}")
    
    xy_e = xy[[ni, nj, nk], :]

    #print(f"xy_e = {xy_e}")
    
    ke, fe = cst_planestress(xy_e, properties)
    
    #print(f"ke = {ke}")
    
    d = [2*ni, 2*ni+1, 2*nj, 2*nj+1, 2*nk, 2*nk+1] # global DOFs from local dofs

    #Direct stiffnes method
    for i in range(6) :
        p = d[i]
        for j in range(6) :
            q = d[j]
            K[p, q] += ke[i,j]
        f[p] += fe[i]
        
fixed_nodes = np.unique(fixed_nodes)    

constrained_DOFs = []
for n in fixed_nodes:
    constrained_DOFs += [2*n, 2*n+1]

print(f"fixed_nodes = {fixed_nodes}")   
print(f"constrained_DOFs = {constrained_DOFs}")

free_DOFs = np.arange(NDOFs)
free_DOFs = np.setdiff1d(free_DOFs, constrained_DOFs)
print(f"free_DOFS = {free_DOFs}")

import matplotlib.pylab as plt

plt.matshow(K)
plt.show()

Kff = K[ix_(free_DOFs, free_DOFs)]
Kfc = K[ix_(free_DOFs, constrained_DOFs)]
Kcf = K[ix_(constrained_DOFs, free_DOFs)]
Kcc = K[ix_(constrained_DOFs, constrained_DOFs)]

ff = f[free_DOFs]
fc = f[constrained_DOFs]

# Solve:
u = zeros((NDOFs,1))

u[free_DOFs] = solve(Kff, ff)

# Get reaction forces:

R = Kcf @ u[free_DOFs] + Kcc @ u[constrained_DOFs] -fc

print(f"Kff = {Kff}")
print(f"R = {R}")

factor = 1e2
uv = u.reshape([-1, 2])

plt.plot(xy[:,0] + factor*uv[:,0], xy[:,1] + factor*uv[:,1],".")

for e in Triangles:
    ni = conec[e,0]
    nj = conec[e,1]
    nk = conec[e,2]
    
    xy_e = xy[[ni, nj, nk, ni], :] + factor*uv[[ni, nj, nk, ni], :]
    plt.plot(xy_e[:,0], xy_e[:,1], "k")

plt.axis("equal")    
plt.show()


from gmsh_post import write_node_data, write_node_data_2, write_element_data

nodes = np.arange(1, Nnodes+1)
write_node_data("ux.msh", nodes, uv[:,0], "Despl. X")
write_node_data("uy.msh", nodes, uv[:,1], "Despl. Y")
write_node_data_2("desplazamientos.msh", nodes, uv[:,0], uv[:,1], "Despl.")



#Calculo de tensiones

sigma_xx = np.zeros(Ntriangles+1)
sigma_yy = np.zeros(Ntriangles+1)
sigma_xy = np.zeros(Ntriangles+1)

i = 0
for e in Triangles:
    ni = conec[e,0]
    nj = conec[e,1]
    nk = conec[e,2]
    
    xy_e = xy[[ni, nj, nk, ni], :]
    
    uv_e = uv[[ni, nj, nk], :]
    
    u_e = uv_e.reshape((-1))
    
    epsilon_e, sigma_e = cst_planestress_post(xy_e, u_e, properties)
    
    sigma_xx[i] = sigma_e[0]
    sigma_yy[i] = sigma_e[1]
    sigma_xy[i] = sigma_e[2]
    
    i += 1
   
elementos = np.array(Triangles)+1
write_element_data("sigma_x.msh", elementos, sigma_xx, "Sigma_x")

print(f"u_9 = {u[2*2+1]}")
print(max(sigma_xx))
#u_y_i = (nodoi*2+1) despl y. del nodo i
