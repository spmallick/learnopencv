mkdir models
wget https://github.com/richzhang/colorization/blob/master/colorization/resources/pts_in_hull.npy?raw=true -O ./pts_in_hull.npy
wget https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt -O ./models/colorization_deploy_v2.prototxt
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ./models/colorization_release_v2.caffemodel
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2_norebal.caffemodel -O ./models/colorization_release_v2_norebal.caffemodel
