// declare Mat variables, thr, gray and src
Mat thr, gray, src;
 
// convert image to grayscale
cvtColor( src, gray, COLOR_BGR2GRAY );
 
// convert grayscale to binary image
threshold( gray, thr, 100,255,THRESH_BINARY );
 
// find moments of the image
Moments m = moments(thr,true);
Point p(m.m10/m.m00, m.m01/m.m00);
 
// coordinates of centroid
cout<< Mat(p)<< endl;
 
// show the image with a point mark at the centroid
circle(src, p, 5, Scalar(128,0,0), -1);
imshow("Image with center",src);
waitKey(0);