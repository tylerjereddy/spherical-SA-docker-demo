'''Demonstrate proposed scipy spherical polygon surface area calculation.

Authors: Tyler Reddy and Edward Edmondson'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import numpy as np
import scipy
import scipy.spatial
import math
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.spherical import SphericalPolygon

def convert_cartesian_array_to_spherical_array(coord_array,angle_measure='radians'):
    '''Take shape (N,3) cartesian coord_array and return an array of the same shape in spherical polar form (r, theta, phi). Based on StackOverflow response: http://stackoverflow.com/a/4116899
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    spherical_coord_array = np.zeros(coord_array.shape)
    xy = coord_array[...,0]**2 + coord_array[...,1]**2
    spherical_coord_array[...,0] = np.sqrt(xy + coord_array[...,2]**2)
    spherical_coord_array[...,1] = np.arctan2(coord_array[...,1], coord_array[...,0])
    spherical_coord_array[...,2] = np.arccos(coord_array[...,2] / spherical_coord_array[...,0])
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = np.degrees(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = np.degrees(spherical_coord_array[...,2])
    return spherical_coord_array

def convert_spherical_array_to_cartesian_array(spherical_coord_array,angle_measure='radians'):
    '''Take shape (N,3) spherical_coord_array (r,theta,phi) and return an array of the same shape in cartesian coordinate form (x,y,z). Based on the equations provided at: http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    cartesian_coord_array = np.zeros(spherical_coord_array.shape)
    #convert to radians if degrees are used in input (prior to Cartesian conversion process)
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = np.deg2rad(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = np.deg2rad(spherical_coord_array[...,2])
    #now the conversion to Cartesian coords
    cartesian_coord_array[...,0] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,1] = spherical_coord_array[...,0] * np.sin(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,2] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,2])
    return cartesian_coord_array

def calculate_haversine_distance_between_spherical_points(cartesian_array_1,cartesian_array_2,sphere_radius):
    '''Calculate the haversine-based distance between two points on the surface of a sphere. Should be more accurate than the arc cosine strategy. See, for example: http://en.wikipedia.org/wiki/Haversine_formula'''
    spherical_array_1 = convert_cartesian_array_to_spherical_array(cartesian_array_1)
    spherical_array_2 = convert_cartesian_array_to_spherical_array(cartesian_array_2)
    lambda_1 = spherical_array_1[1]
    lambda_2 = spherical_array_2[1]
    phi_1 = spherical_array_1[2]
    phi_2 = spherical_array_2[2]
    #we rewrite the standard Haversine slightly as long/lat is not the same as spherical coordinates - phi differs by pi/4
    spherical_distance = 2.0 * sphere_radius * math.asin(math.sqrt( ((1 - math.cos(phi_2-phi_1))/2.) + math.sin(phi_1) * math.sin(phi_2) * ( (1 - math.cos(lambda_2-lambda_1))/2.)  ))
    return spherical_distance

def calculate_surface_area_of_a_spherical_Voronoi_polygon(array_ordered_Voronoi_polygon_vertices,sphere_radius):
    '''Calculate the surface area of a polygon on the surface of a sphere. Based on equation provided here: http://mathworld.wolfram.com/LHuiliersTheorem.html
    Decompose into triangles, calculate excess for each'''
    #have to convert to unit sphere before applying the formula
    spherical_coordinates = convert_cartesian_array_to_spherical_array(array_ordered_Voronoi_polygon_vertices)
    spherical_coordinates[...,0] = 1.0
    array_ordered_Voronoi_polygon_vertices = convert_spherical_array_to_cartesian_array(spherical_coordinates)
    n = array_ordered_Voronoi_polygon_vertices.shape[0]
    #point we start from
    root_point = array_ordered_Voronoi_polygon_vertices[0]
    totalexcess = 0
    #loop from 1 to n-2, with point 2 to n-1 as other vertex of triangle
    # this could definitely be written more nicely
    b_point = array_ordered_Voronoi_polygon_vertices[1]
    root_b_dist = calculate_haversine_distance_between_spherical_points(root_point, b_point, 1.0)
    for i in 1 + np.arange(n - 2):
        a_point = b_point
        b_point = array_ordered_Voronoi_polygon_vertices[i+1]
        root_a_dist = root_b_dist
        root_b_dist = calculate_haversine_distance_between_spherical_points(root_point, b_point, 1.0)
        a_b_dist = calculate_haversine_distance_between_spherical_points(a_point, b_point, 1.0)
        s = (root_a_dist + root_b_dist + a_b_dist) / 2.
        totalexcess += 4 * math.atan(math.sqrt( math.tan(0.5 * s) * math.tan(0.5 * (s-root_a_dist)) * math.tan(0.5 * (s-root_b_dist)) * math.tan(0.5 * (s-a_b_dist))))
    return totalexcess * (sphere_radius ** 2)

def plot_ref_unit_sphere(ax):
    # plot the unit sphere for reference 
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=5, cstride=5, color='y', alpha=0.1)
    ax.azim = 90
    ax.elev = 30
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def perform_sample_calculation_and_plot(spherical_polygon_vertices, fraction_unit_sphere_area_expected, order, output_file):
    output_file.write('input spherical polygon vertices:' + str(spherical_polygon_vertices) + '\n')
    #the surface area of a sphere is given by 4 * math.pi * r ** 2
    unit_sphere_surface_area = 4 * math.pi
    expected_SA = unit_sphere_surface_area * fraction_unit_sphere_area_expected
    output_file.write('expected_SA for {fraction} fraction of unit sphere:'.format(fraction=fraction_unit_sphere_area_expected) + str(expected_SA) + '\n')
    calculated_SA = calculate_surface_area_of_a_spherical_Voronoi_polygon(spherical_polygon_vertices,1.0)
    output_file.write('calculated_SA for {fraction} fraction of unit sphere:'.format(fraction=fraction_unit_sphere_area_expected) + str(calculated_SA) + '\n')
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection = '3d')
    plot_ref_unit_sphere(ax)
    #old style matplotlib spherical polygon
    polygon = Poly3DCollection([spherical_polygon_vertices], alpha=1.0)
    polygon.set_color('blue')
    ax.add_collection3d(polygon)
    fig.savefig('{fraction}_plot_old_order_{order}.png'.format(fraction=fraction_unit_sphere_area_expected, order=order), dpi = 300)
    #new style matplotlib spherical polygon (Nikolai's branch)
    fig2 = matplotlib.pyplot.figure()
    ax = fig2.add_subplot(111, projection = '3d')
    plot_ref_unit_sphere(ax)
    polygon = SphericalPolygon(spherical_polygon_vertices) 
    polygon.add_to_ax(ax, alpha=1.0, color='blue')
    fig2.savefig('{fraction}_plot_new_{order}.png'.format(fraction=fraction_unit_sphere_area_expected, order=order), dpi = 300)

if __name__ == '__main__':

    with open('log.txt', 'w') as output_file:

        one_eighth_polygon_vertices = np.array([[0, 1., 0], [0, 0, 1.], [-1., 0, 0]]) 
        perform_sample_calculation_and_plot(spherical_polygon_vertices = one_eighth_polygon_vertices, fraction_unit_sphere_area_expected = 1./8., order=1, output_file=output_file)

        one_eighth_polygon_vertices_2 = np.array([[1,0,0],[0,0,1],[0,-1,0]]) #Nikolai's plotting code doesn't plot this set of vertices for some reason
        perform_sample_calculation_and_plot(spherical_polygon_vertices = one_eighth_polygon_vertices_2, fraction_unit_sphere_area_expected = 1./8., order=2, output_file=output_file)

        three_eighths_polygon_vertices = np.array([[0,0,1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]) #sensitive to specified order!
        perform_sample_calculation_and_plot(spherical_polygon_vertices = three_eighths_polygon_vertices, fraction_unit_sphere_area_expected = 3./8., order=1, output_file=output_file)

        three_eighths_polygon_vertices_2 = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]]) #sensitive to specified order!
        perform_sample_calculation_and_plot(spherical_polygon_vertices = three_eighths_polygon_vertices_2, fraction_unit_sphere_area_expected = 3./8., order=2, output_file=output_file)

        #I can't get the surface area calculation to work for a hemisphere, but then how does one provide the 'ordered' vertices for a hemisphere??
