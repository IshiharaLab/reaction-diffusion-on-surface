"""
- Solve Reaction-Diffusion equations of Brusselator on an spherical surface, see Ref.[1] for details.
- Polar coordinate
- Surface shape is given by a radial function R(theta,phi) = R + k( cos(2theta)-1)cos(theta) .
  When k = 0, the surface is a perfect sphere.
- Surface discretization by triangular mesh: spherical_mesh.py.
- Laplace-Beltrami operator follows [2]
- Explicit Euler scheme for update.
- Parameters (you can change them in the section "model parameters").
    Parameters for the reaction diffusion equations: a, b, Du, Dv
    Parameters for the surface shape : N, R, k   (N controls the size of mesh)

    1. Ryosuke Nishide and Shuji Ishihara
    "Pattern Propagation Driven by Surface Curvature" (2021)

    2. G. Xu
    "Discrete Laplace-Beltrami operator on sphere and optimal spherical triangulations"
    Int. J. Comput. Geom. Appl. 16, p. 75--93 (2006).

"""

import time
import numpy as np
import scipy.sparse as sparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import spherical_mesh

"""
model, reaction part
"""
def Brusselator(U,V,fU,fV,a,b):
    fV[:] = b * U[:] - U[:]*U[:]*V[:]
    fU[:] = a - U[:] - fV[:]


"""
model parameters
"""
## for reaction-diffusion
a, b, Du, Dv = 2.0, 4.5, 0.5, 1.8
## for surface;  add points times : N, radius : R, deformed sphere parameter : k
N, R, k = 5, 6, 1.0


"""make mesh & Laplace-Beltrami operator"""
n, simplices, points_3d = spherical_mesh.spherical_mesh_generator(N, R, k)
Lap = spherical_mesh.Laplacian(n, simplices, points_3d)

Lap_csr = sparse.csr_matrix(Lap)


""" time """
dt = 1.0e-3
Time, STEP = int(2e8), int(1e6)   # simulation time , steps b/w output

""" arrays """
U_time, V_time = np.zeros((int(Time/STEP),n)), np.zeros((int(Time/STEP),n))
fU, fV = np.zeros(n), np.zeros(n)

""" initial_condition """
U, V = a * np.ones(n), (b/a)*np.ones(n) + 0.1*(np.random.rand(n)-0.5)


""" figure, initial setting """
fig3d = plt.figure()
ax3d  = fig3d.add_subplot(111, projection='3d')
minu, maxu = 1/5*a, 2*a

def scatter_plot(U_time,t):
    ax3d.cla()
    ax3d.scatter3D( points_3d[:,0], points_3d[:,1], points_3d[:,2],
                   c=U_time[t,:], vmin=minu, vmax=maxu, s=20 )
    ax3d.set_xlim(-R,R), ax3d.set_ylim(-R,R), ax3d.set_zlim(-R,R)
    ax3d.set_box_aspect((1,1,1))
    ax3d.set_title('time= %.2f' % (t*STEP*dt,))

""" time evolution """
count=0
for tt in range(Time):

    if tt%STEP==0:
        U_time[count,:], V_time[count,:] = U[:], V[:]
        scatter_plot(U_time, count)
        plt.pause(0.0001)
        count+=1

    if tt%100000==0:
        print('%.2f %f %f' % (tt*dt,np.max(U),np.min(U),)) # for check

    Brusselator(U,V,fU,fV,a,b)
    U[:] += dt*( Du*Lap_csr.dot(U)[:] + fU[:] )
    V[:] += dt*( Dv*Lap_csr.dot(V)[:] + fV[:] )

np.savez_compressed('Brusselator_spherical_k='+str(k), U_time, V_time, simplices, points_3d)
