'''Demonstrate proposed scipy spherical polygon surface area calculation.

Authors: Tyler Reddy and Edward Edmondson'''

import matplotlib
import matplotlib.pyplot
import numpy as np
import math

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

if __name__ == '__main__':

    #the surface area of a sphere is given by 4 * math.pi * r ** 2
    unit_sphere_surface_area = 4 * math.pi

    #calculate the surface area of a spherical polygon covering 1/8th the surface area of the unit sphere
    #compare with expected value
    expected_one_eighth_SA = unit_sphere_surface_area / 8.
    one_eighth_polygon_vertices = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    calculated_one_eighth_SA = calculate_surface_area_of_a_spherical_Voronoi_polygon(one_eighth_polygon_vertices,1.0)
    print 'expected_one_eighth_SA:', expected_one_eighth_SA
    print 'calculated_one_eighth_SA:', calculated_one_eighth_SA
    
    
    #repeat for 3/8ths of a unit sphere
    expected_three_eighth_SA = unit_sphere_surface_area * (3./8.)
    three_eighth_polygon_vertices = np.array([[0,0,1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]) #sensitive to specified order!
    calculated_three_eighth_SA = calculate_surface_area_of_a_spherical_Voronoi_polygon(three_eighth_polygon_vertices,1.0)
    print 'expected_three_eighth_SA:', expected_three_eighth_SA
    print 'calculated_three_eighth_SA:', calculated_three_eighth_SA

    #I can't get the surface area calculation to work for a hemisphere, but then how does one provide the 'ordered' vertices for a hemisphere??
