#! /usr/bin/env python

from dynamic_reconfigure.parameter_generator_catkin import bool_t
from dynamic_reconfigure.parameter_generator_catkin import double_t
from dynamic_reconfigure.parameter_generator_catkin import int_t
from dynamic_reconfigure.parameter_generator_catkin import str_t

# set up parameters that we care about
PACKAGE = 'pcl_ros'


def add_common_parameters(gen):
    # add(self, name, paramtype, level, description, default = None, min = None,
    # max = None, edit_method = '')
    gen.add('max_iterations', int_t, 0,
            'The maximum number of iterations the algorithm will run for',
            50, 0, 100000)
    gen.add('probability', double_t, 0,
            'The desired probability of choosing at least one sample free from outliers',
            0.99, 0.5, 0.99)
    gen.add('distance_threshold', double_t, 0,
            'The distance to model threshold',
            0.02, 0, 1.0)
    gen.add('optimize_coefficients', bool_t, 0,
            'Model coefficient refinement',
            True)
    gen.add('radius_min', double_t, 0,
            'The minimum allowed model radius (where applicable)',
            0.0, 0, 1.0)
    gen.add('radius_max', double_t, 0,
            'The maximum allowed model radius (where applicable)',
            0.05, 0, 1.0)
    gen.add('eps_angle', double_t, 0,
            ('The maximum allowed difference between the model normal '
             'and the given axis in radians.'),
            0.17, 0.0, 1.5707)
    gen.add('min_inliers', int_t, 0,
            'The minimum number of inliers a model must have in order to be considered valid.',
            0, 0, 100000)
    gen.add('input_frame', str_t, 0,
            ('The input TF frame the data should be transformed into, '
             'if input.header.frame_id is different.'),
            '')
    gen.add('output_frame', str_t, 0,
            ('The output TF frame the data should be transformed into, '
             'if input.header.frame_id is different.'),
            '')
