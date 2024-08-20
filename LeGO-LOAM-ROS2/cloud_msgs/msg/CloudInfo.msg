std_msgs/Header header

int32[] start_ring_index
int32[] end_ring_index

float32 start_orientation
float32 end_orientation
float32 orientation_diff

# true - ground point, false - other points
bool[]    segmented_cloud_ground_flag
# point column index in range image
uint32[]  segmented_cloud_col_ind
# point range
float32[] segmented_cloud_range
