import numpy as np
import scipy.spatial as ss

def spherical_mesh_generator(N, R, k):
    """ regular_icosahedron """
    G = ( 1+np.sqrt(5) ) / 2
    r = np.sqrt( 1+G**2 )
    icosahedron = R / r * np.array([ [1, G, 0], [-1, -G, 0], [-1, G, 0], [1, -G, 0],
                                     [0, 1, G], [0, -1, -G], [0, -1, G], [0, 1, -G],
                                     [G, 0, 1], [-G, 0, -1], [G, 0, -1], [-G, 0, 1] ])

    points_3d = np.copy(icosahedron)
    n = len(points_3d)

    CH = ss.ConvexHull(points_3d)
    simplices1 = CH.simplices

    """ add points """
    for ii in range(N):
        new_points = []
        print( 'add points:', ii+1, 'times' )
        for i in range( len(simplices1[:,0]) ):
            """vertex"""
            v0, v1, v2 = simplices1[i,0], simplices1[i,1], simplices1[i,2]

            new_points.append([ (points_3d[v0,0] + points_3d[v1,0])/2,
                                (points_3d[v0,1] + points_3d[v1,1])/2,
                                (points_3d[v0,2] + points_3d[v1,2])/2 ])

            new_points.append([ (points_3d[v2,0] + points_3d[v1,0])/2,
                                (points_3d[v2,1] + points_3d[v1,1])/2,
                                (points_3d[v2,2] + points_3d[v1,2])/2 ])

            new_points.append([ (points_3d[v0,0] + points_3d[v2,0])/2,
                                (points_3d[v0,1] + points_3d[v2,1])/2,
                                (points_3d[v0,2] + points_3d[v2,2])/2 ])

        new_points = np.array( list(map(list, set(map(tuple, new_points)))) )

        nn = len(new_points)

        """ sphere coordinate """
        x, y, z = np.zeros(nn), np.zeros(nn), np.zeros(nn)
        THETA, PHI = np.zeros(nn), np.zeros(nn)

        THETA[:] = np.arccos(new_points[:,2] / np.linalg.norm(new_points[:,:], axis=1, ord=2))
        PHI[:]= np.arctan2(new_points[:,1], new_points[:,0]) + np.pi

        x[:] = R * np.cos(PHI[:]) * np.sin(THETA[:])
        y[:] = R * np.sin(PHI[:]) * np.sin(THETA[:])
        z[:] = R * np.cos(THETA[:])

        new_points = np.array( [ [x[k], y[k], z[k]] for k in range(nn) ] )

        points_3d = np.vstack( [points_3d, new_points] )

        """ConvexHull"""
        CH = ss.ConvexHull(points_3d)
        simplices1  = CH.simplices

    n = len(points_3d)

    """ rotation """
    theta_x = 1e-4
    Rx = np.array([ [1, 0, 0],
                    [ 0, np.cos(theta_x), np.sin(theta_x) ],
                    [ 0,-np.sin(theta_x), np.cos(theta_x) ] ])

    for i in range(n):
        points_3d[i,:] = np.dot(Rx[:,:], points_3d[i,:])

    """ 2d_projection """
    X, Y = np.zeros(n), np.zeros(n)
    X[:] = 2 * R * points_3d[:,0] / (R - points_3d[:,2])
    Y[:] = 2 * R * points_3d[:,1] / (R - points_3d[:,2])

    points = np.array( [[X[k], Y[k]] for k in range(n)] )

    tri    = ss.Delaunay(points, furthest_site = False)
    tri2   = ss.Delaunay(points, furthest_site = True)

    """ Delaunay """
    simplices = np.vstack([tri.vertices, tri2.vertices])

    """ deformed sphere coordinate """
    x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
    THETA, PHI = np.zeros(n), np.zeros(n)

    THETA[:] = np.arccos(points_3d[:,2] / np.linalg.norm(points_3d[:,:], axis=1, ord=2))
    PHI[:]   = np.arctan2(points_3d[:,1], points_3d[:,0]) + np.pi

    x[:] = ( R + k * ( np.cos(2*THETA[:])-1) * np.cos(THETA[:]) ) * np.cos(PHI[:]) * np.sin(THETA[:])
    y[:] = ( R + k * ( np.cos(2*THETA[:])-1) * np.cos(THETA[:]) ) * np.sin(PHI[:]) * np.sin(THETA[:])
    z[:] = ( R + k * ( np.cos(2*THETA[:])-1) * np.cos(THETA[:]) ) * np.cos(THETA[:])

    points_3d[:] = np.array([[x[k], y[k], z[k]] for k in range(n)])[:]

    return n, simplices, points_3d

def Laplacian(n, simplices, points_3d):

    Lap = np.zeros( (n, n) )

    array_length   = simplices.size
    vertex         = np.zeros( (array_length, 3), dtype=int )
    edge_length    = np.zeros( (array_length, 3), dtype=float )
    angle          = np.zeros( (array_length, 3), dtype=float )
    area           = np.zeros( n, dtype=float )

    count=0
    for i in range(n):
        """
        頂点iを含む単体のindex
        index[0]がsimplicesの配列の位置
        index[1]がsimplicesの配列の中の位置
        """
        index = np.where( simplices==i )

        area_seg = 0
        mean_curvature_vector_seg = np.zeros( 3, dtype=float )
        m = 0
        for j in simplices[index[0],:]:

            """sort"""
            j = np.roll(j, -index[1][m])

            """vertex"""
            v0, v1, v2 = j[0], j[1], j[2]
            vertex[count,:] = np.array([v0, v1, v2])

            """edge_length"""
            length0 = np.linalg.norm( points_3d[v1,:] - points_3d[v2,:], ord=2 )
            length1 = np.linalg.norm( points_3d[v2,:] - points_3d[v0,:], ord=2 )
            length2 = np.linalg.norm( points_3d[v0,:] - points_3d[v1,:], ord=2 )
            edge_length[count,:] = np.array([length0, length1, length2])[:]

            """angle"""
            angle0 = np.arccos( np.dot( points_3d[v1,:]-points_3d[v0,:], points_3d[v2,:]-points_3d[v0,:] )/(length1 * length2) )
            angle1 = np.arccos( np.dot( points_3d[v0,:]-points_3d[v1,:], points_3d[v2,:]-points_3d[v1,:] )/(length2 * length0) )
            angle2 = np.arccos( np.dot( points_3d[v0,:]-points_3d[v2,:], points_3d[v1,:]-points_3d[v2,:] )/(length1 * length0) )
            angle[count,:]= np.array([angle0, angle1, angle2])

            """area_segmentation"""
            area_seg += (length1**2) / np.tan( angle1 ) + (length2**2) / np.tan( angle2 )

            m+=1
            count+=1

        """area"""
        area[i] = area_seg/8

    for j in range(array_length):
        v0, v1, v2 = int(vertex[j,0]), int(vertex[j,1]), int(vertex[j,2])
        angle0, angle1, angle2  = angle[j,0], angle[j,1], angle[j,2]

        m1 = np.where( (vertex[:,0]==v0) & (vertex[:,2]==v1) )
        m2 = np.where( (vertex[:,0]==v0) & (vertex[:,1]==v2) )

        angle3, angle4 = angle[m1[0][0],1], angle[m2[0][0],2]

        Lap[v0, v0] += - 1 / area[v0] * ( 1/np.tan( angle1 ) + 1/np.tan( angle2 ) )/2
        Lap[v0, v1] =    1 / area[v0] * ( 1/np.tan( angle2 ) + 1/np.tan( angle3 ) )/2
        Lap[v0, v2] =    1 / area[v0] * ( 1/np.tan( angle1 ) + 1/np.tan( angle4 ) )/2

    return Lap