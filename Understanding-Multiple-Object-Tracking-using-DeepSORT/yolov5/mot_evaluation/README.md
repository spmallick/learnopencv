<!-- ---
export_on_save:
 html: true
--- -->

# MOT2016/2017 Evaluation Toolkit
It is a python implementation of [MOT](https://motchallenge.net/). However, I only reimplement the 2D evaluation part under MOT16 file format.

The IDF1, IDP, IDR now is not agreed with official toolkit. The original implementation might have an indexing bug in the computation of these metrics. 

### Metrics
The metrics of MOT16 are based on the following papers:

1. CLEAR MOT
- Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." Journal on Image and Video Processing 2008 (2008): 1.
2. MTMC
- Ristani, Ergys, et al. "Performance measures and a data set for multi-target, multi-camera tracking." European Conference on Computer Vision. Springer, Cham, 2016.

Typical evaluation format is shown as
```bash
IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
```

<span style="color:red;"> Github doesn't support latex formula. To have a better reading experience see [here](http://htmlpreview.github.io/?https://github.com/shenh10/mot_evaluation/blob/master/README.html)</span>

The meaning of each alias is 
- **IDF1(ID F1 Score)**: 
    $$ IDF1 = \frac{2 * IDTP} {2 * IDTP + IDFP + IDFN} $$
- **IDP(ID Precison)**: 
    $$ IDP = \frac{IDTP}{IDTP + IDFP} $$
- **IDR(ID Recall)**: 
    IDR = \frac{IDTP}{IDTP + IDFN}
- **IDTP**: 
    The longest associated trajectory matching to a groundtruth trajectory is regarded as the gt's true ID.
    Then other trajectories matching to this gt is regarded as a 'IDFP'.
- **Rcll(Recall)**: 
    The ratio of TP boxes to GT boxes.
     $$ Recall = \frac{TP}{TP + FN} $$
- **Prcn(Precision)**: 
    The ratio of TP boxes to all detected boxes.
     $$ Precision = \frac{TP}{TP + FP}$$
- **FAR(False Alarm Ratio)**: 
    The ratio of FP boxes to frame number.
     $$FAR = \frac{FP}{\sum_t 1} $$
- **GT(Number of Groundtruth Trajectory)**: 
    The number of groundtruth trajectories.
- **MT(Number of Mostly Tracked Trajectory)**: 
    The number of trajectories that have over 80% target tracked. 
- **PT(Number of Partially Tracked Trajectory)**: 
    The number of trajectories that have 20% to 80% target tracked.
     $$ PT = GT - MT - ML $$
- **ML(Number of Mostly Lost Trajectory)**: 
    The number of trajectories that have less than 20% target tracked.
    Total false positive number among all frames.
     $$ FP = \sum_t \sum_i {fp}_{i, t} $$
- **FN(Number of False Negatives)**: 
    Total false negative number among all frames.
     $$ FN = \sum_t \sum_i fn_{i, t} $$
- **IDs(Number of IDSwitch)**:
    ID switch number, indicating the times of ID jumps.
     $$ IDs = \sum_t ids_{i, t}$$ 
- **FM(Number of Fragmentations)**:
    ID switch is the special case of fragmentation when ID jumps.
     Fragmentation reflects the continousity of trajectories.
     When trajectories are determinated, it counts all missed target in each frame.
- **MOTA(Number of Multiple Object Tracking Accuracy)**:
    A metric reflects the tracking accuracy. It has intergrated consideration of FN, FP, and IDS.
     $$ T = \sum_t \sum_i gt_{i,t} $$, $$ MOTA = 1 - \frac{FN + FP + IDS}{T}$$
- **MOTP(Number of Multiple Object Tracking Precision)**:
     A metric reflects the tracking precision.
      $$ MOTP =  \frac{\sum_{i,t}IoU_{t, i}}{TP} $$
- **MOTAL(MOTA Log)**
    $$MOTA = 1 - \frac{FN + FP + \log_{10}IDS}{T}$$


### To Do
- Supporting MOT15/MOT17 and DukeMTMC file format 
- Optimize the implemenentaion. The current implementation is slow. Any contribution to speed up the code is welcomed.
