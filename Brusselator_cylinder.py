"""
- Solve Reaction-Diffusion equations of Brusselator on an axisymetric surface.
- Cylindrical coordinate with periodic boundary condition.
- Surface shape is given by a radial function r(x) = d + k1 cos(x) + k2 cos(2x-gamma pi/2).
  When k1 = 0 and k2 = 0, the surface is a simple cylinder.
- Laplace-Beltrami operator is discretized by the finite differential method.
- Explicit Euler scheme for time evolution.
- Parameters (you can change them in the section "model parameters").
    Parameters for the reaction diffusion equations: a, b, Du, Dv
    Parameters for the surface shape : d, k1, k2, gamma

See [1] for details.
    1. Ryosuke Nishide and Shuji Ishihara
    "Pattern Propagation Driven by Surface Curvature" (2022)

"""

import numpy as np
import scipy.sparse as sparse
import matplotlib
import matplotlib.pyplot as plt


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
## for surface
d, k1, k2 = 1.7, 0.3, 0.05
gamma = 0.0


""" time """
dt = 1.0e-4
Time, STEP = int(2e8), int(1e6)   # simulation time , steps b/w output


""" grid size """
Nx,  Ntheta  = 100,     100
Lx,  Ltheta  = 4*np.pi, 2*np.pi
dx,  dtheta  = Lx/Nx,   Ltheta/Ntheta
dx2, dtheta2 = dx**2,   dtheta**2
X,   THETA   = np.arange(Nx)*dx, np.arange(Ntheta)*dtheta


""" surface """
X2  = np.arange(Nx*2)*(dx/2)
cos1, cos2 = k1 * np.cos(X2), k2 * np.cos( 2*X2 - gamma * np.pi/2 )
sin1, sin2 = k1 * np.sin(X2), k2 * np.sin( 2*X2 - gamma * np.pi/2 )

r      = d + ( cos1 + cos2 )
drdx   =   - ( sin1 + 2 * sin2 )


""" functions associated with metric """
## M=f/sqrt(1+(dfdx)^2) ,GI=1/det(g)=1/f*sqrt(1+(dfdx)^2),F=f[x,y]
## ME:M_even, MO:M_odd
ME, MO, GIE, R = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
ME[:]  = r[::2]  / np.sqrt( 1+(drdx[::2]**2) )
MO[:]  = r[1::2] / np.sqrt( 1+(drdx[1::2]**2) )
GIE[:] = 1 / ( r[::2] * np.sqrt(1+(drdx[::2]**2)) )
R[:]   = r[::2]


""" Laplace-Beltrami operator """
Lap = sparse.lil_matrix( (Nx * Ntheta, Nx * Ntheta) )
for x in range(Nx):
    for theta in range(Ntheta):
        Lap[x+Nx*theta, x+Nx*theta] = - GIE[x]*(MO[x]+MO[x-1])/dx2 - 2/(R[x]**2)/dtheta2

        Lap[x+Nx*theta, Nx*theta+(x-1)%Nx] = GIE[x]*MO[(x-1)%Nx]/dx2
        Lap[x+Nx*theta, Nx*theta+(x+1)%Nx] = GIE[x]*MO[x]/dx2

        Lap[x+Nx*theta, x+Nx*((theta-1)%Ntheta)] = 1/(R[x]**2)/dtheta2
        Lap[x+Nx*theta, x+Nx*((theta+1)%Ntheta)] = 1/(R[x]**2)/dtheta2

Lap_csr = Lap.tocsr()

# erase memories unsed below
del Lap, ME, MO, GIE, R, X2, sin1, sin2, cos1, cos2, drdx



"""
main
"""

""" arrays """
U_time, V_time = np.zeros((int(Time/STEP),Nx * Ntheta)), np.zeros((int(Time/STEP),Nx * Ntheta))
fU, fV = np.zeros(Nx * Ntheta), np.zeros(Nx * Ntheta)

""" initial_condition """
U, V = a * np.ones((Nx*Ntheta), float), (b/a)*np.ones((Nx*Ntheta), float) + 0.1*(np.random.rand(Nx*Ntheta)-0.5)


""" figure, initial setting """
fig3d = plt.figure()
ax3d  = fig3d.add_subplot(111, projection='3d')
minu, maxu = 1/4*a, 2*a

x3D, theta3D = np.arange(0, Nx)*dx-2*np.pi, np.arange(0, Ntheta)*dtheta
X3D, THETA3D = np.meshgrid(x3D, theta3D)

cos1, cos2 = k1 * np.cos(X3D), k2 * np.cos( 2*X3D - gamma * np.pi/2 )
sin1, sin2 = k1 * np.sin(X3D), k2 * np.sin( 2*X3D - gamma * np.pi/2 )

R3D = d + ( cos1 + cos2 )
Y3D, Z3D = R3D * np.cos(THETA3D), R3D * np.sin(THETA3D)

norm  = matplotlib.colors.Normalize(vmin=minu, vmax=maxu)
cm    = plt.get_cmap('viridis')

def surface_plot(U_time,t):
    ax3d.cla()
    colors = cm(norm(U_time[t,:].reshape( Ntheta, Nx )))
    u3d = ax3d.plot_surface(X3D, Y3D, Z3D, shade=False,
                            facecolors = colors)
    ax3d.set_xlim(-2*np.pi,2*np.pi), ax3d.set_ylim(-2*np.pi,2*np.pi), ax3d.set_zlim(-2*np.pi,2*np.pi)
    ax3d.set_title('time= %.2f' % (t*STEP*dt,))


""" time evolution """
count=0
for tt in range(Time):

    if tt%STEP==0:
        U_time[count,:], V_time[count,:] = U[:], V[:]
        surface_plot(U_time, count)
        plt.pause(0.0001)
        count+=1

    if tt%100000==0:
        print('%.2f %f %f' % (tt*dt,np.max(U),np.min(U),)) # for check

    Brusselator(U,V,fU,fV,a,b)
    U[:] += dt*( Du*Lap_csr.dot(U)[:] + fU[:] )
    V[:] += dt*( Dv*Lap_csr.dot(V)[:] + fV[:] )
    
np.savez_compressed('Brusselator_cyliderical_gamma='+str(gamma), U_time, V_time)
