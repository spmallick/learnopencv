## Center of Blob using Python and C++

To run the code to find center of a single blob, run the following commands:-

For python:-

`python3 single_blob.py --ipimage image_name`


For C++:-

1. ``g++ single_blob.cpp `pkg-config opencv --cflags --libs` -o output``

2. `./output image_name`

To run the code to find center of multiple blobs, run the following commands:-

For python:-

`python3 center_of_multiple_blob.py --ipimage image_name`

For C++:-

1. ``g++ center_of_multiple_blob.cpp `pkg-config opencv --cflags --libs` -o output``

2. `./output image_name`

