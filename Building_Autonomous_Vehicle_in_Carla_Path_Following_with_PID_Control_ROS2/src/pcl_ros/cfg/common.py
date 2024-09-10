#! /usr/bin/env python

from dynamic_reconfigure.parameter_generator_catkin import bool_t
from dynamic_reconfigure.parameter_generator_catkin import double_t
from dynamic_reconfigure.parameter_generator_catkin import str_t

# set up parameters that we care about
PACKAGE = 'pcl_ros'


def add_common_parameters(gen):
    # def add (self, name, paramtype, level, description, default = None, min = None,
    # max = None, edit_method = ''):
    gen.add('filter_field_name', str_t, 0, 'The field name used for filtering', 'z')
    gen.add('filter_limit_min', double_t, 0,
            'The minimum allowed field value a point will be considered from',
            0.0, -100000.0, 100000.0)
    gen.add('filter_limit_max', double_t, 0,
            'The maximum allowed field value a point will be considered from',
            1.0, -100000.0, 100000.0)
    gen.add('filter_limit_negative', bool_t, 0,
            ('Set to true if we want to return the data outside '
             '[filter_limit_min; filter_limit_max].'),
            False)
    gen.add('keep_organized', bool_t, 0,
            ('Set whether the filtered points should be kept and set to NaN, '
             'or removed from the PointCloud, thus potentially breaking its organized structure.'),
            False)
    gen.add('input_frame', str_t, 0,
            ('The input TF frame the data should be transformed into before processing, '
             'if input.header.frame_id is different.'),
            '')
    gen.add('output_frame', str_t, 0,
            ('The output TF frame the data should be transformed into after processing, '
             'if input.header.frame_id is different.'),
            '')
