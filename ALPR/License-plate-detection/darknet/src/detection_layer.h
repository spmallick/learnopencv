#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer detection_layer;

#ifdef __cplusplus
extern "C" {
#endif
detection_layer make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
void forward_detection_layer(const detection_layer l, network_state state);
void backward_detection_layer(const detection_layer l, network_state state);
void get_detection_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network_state state);
void backward_detection_layer_gpu(detection_layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
