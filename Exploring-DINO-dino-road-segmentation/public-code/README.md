# Scene Understanding Challenge for Autonomous Navigation in Unstructured Environments

Code for working with the dataset used for the [Scene Understanding Challenge for Autonomous Navigation in Unstructured Environments](http://cvit.iiit.ac.in/scene-understanding-challenge-2018/). For details of getting the dataset and updates see:
- https://cvit.iiit.ac.in/autonue2021/
- https://cvit.iiit.ac.in/autonue2019/
- http://cvit.iiit.ac.in/autonue2018/
- http://cvit.iiit.ac.in/scene-understanding-challenge-2018/ 


# AutoNUE 2021 (Domain Adaptation and Semantic Segmentation)

This repository contains the datasets related to domain adaptation and segmentation challenges for AutoNEU 2021, CVPR Workshop. For more details, please visit https://cvit.iiit.ac.in/autonue2021/challenge. For the segmentation challenge, please skip "Source datasets" section below.


## Source datasets:

Participants are requested to download the datasets from original websites, given below for easy reference:-
1. https://www.mapillary.com/dataset/vistas?pKey=q0GhQpk20wJm1ba1mfwJmw
2. https://bdd-data.berkeley.edu/ (you might have to click on Advanced tab, and then click on "proceed to bdd-data.berkeley.edu")
3. https://download.visinf.tu-darmstadt.de/data/from_games/
4. https://www.cityscapes-dataset.com/examples/#fine-annotations (only fine-annotations to be used)

After downloading all the source datasets, move them to folder ./domain_adaptation/source/datasets/. Its folder structure should be as follows:
```
datasets
  |--mapillary-vistas-dataset_public_v1.1/
  |  |--training/
  |  |  |--images/
  |  |  |--labels/
  |  |--validation/
  |  |  |--images/
  |  |  |--labels/
  |  |--testing/
  |     |--images/
  |--bdd100k/
  |  |--seg/
  |     |--images/
  |     |  |--train/
  |     |  |--val/
  |     |  |--test/
  |     |--labels/
  |        |--train/
  |        |--val/
  |--gta/
  |  |--images/
  |  |--labels/
  |--cityscapes/
     |--gtFine/
     |  |--train/
     |  |--val/
     |  |--test/
     |--leftImg8bit/
        |--train/
        |--val/
        |--test/
```


Run the following commands:

```
pip3 install requirements.txt
chmod +x domain_adaptation/source/prep_all.sh
./domain_adaptation/source/prep_all.sh
```

This will create a folder "domain_adaptation/source/source_datasets_dir/" where you will find the images and annotations for the source dataset to be used for any of the domain adaptation challenges.

## Target datasets:

**For using first add helpers/ to $PYTHONPATH**
```
export PYTHONPATH="${PYTHONPATH}:helpers/"
```

### Dataset Structure 

The structure is similar to the cityscapes dataset. That is:
```
gtFine/{split}/{drive_no}/{img_id}_gtFine_polygons.json for ground truths
leftImg8bit/{split}/{drive_no}/{img_id}_leftImg8bit.png for image frames
```
#### Semantic Segmentation

Furthermore for training, label masks needs to be generated as described below resulting in the following files:
```
gtFine/{split}/{drive_no}/{img_id}_gtFine_labellevel3Ids.png
gtFine/{split}/{drive_no}/{img_id}_gtFine_instancelevel3Ids.png
```
### Labels

See helpers/anue_labels.py

#### Generate Label Masks (for training/evaluation) (Semantic/Instance/Panoptic Segmentation)
```bash
python preperation/createLabels.py --datadir $ANUE --id-type $IDTYPE --color [True|False] --instance [True|False] --num-workers $C
```

- ANUE is the path to the AutoNUE dataset
- IDTYPE can be id, csId, csTrainId, level3Id, level2Id, level1Id. 
- color True  generates the color masks
- instance True generates the instance masks with the id given by IDTYPE
- panoptic True generates panoptic masks in the format similar to COCO. See the modified evaluation scripts here: https://github.com/AutoNUE/panopticapi
- C is the number of threads to run in parallel


For the supervised domain adaptation and semantic segmentation tasks, the masks should be generated using IDTYPE of level3Id and used for training models (similar to trainId in cityscapes). This can be done by the command:
```bash
python preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers $C
```

Following commands are updated for the target labels of other domain adaptation tasks:

```
python3 preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers $C --semisup_da True
python3 preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers $C --weaksup_da True
python3 preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers $C --unsup_da True

```
The bounding box labels for weakly supervised domain adapation can be downloaded from here: https://github.com/AutoNUE/public-code/tree/master/domain_adaptation/target/weakly-supervised


The generated files:

- _gtFine_labelLevel3Ids.png will be used for semantic segmentation



# AutoNUE 2019

**For using first add helpers/ to $PYTHONPATH**
```
export PYTHONPATH="${PYTHONPATH}:helpers/"
```

**The code has been tested on python 3.6.4**

## Dataset Structure 

The structure is similar to the cityscapes dataset. That is:
```
gtFine/{split}/{drive_no}/{img_id}_gtFine_polygons.json for ground truths
leftImg8bit/{split}/{drive_no}/{img_id}_leftImg8bit.png for image frames
```
### Semantic Segmentation and Instance Segmentation

Furthermore for training, label masks needs to be generated as described below resulting in the following files:
```
gtFine/{split}/{drive_no}/{img_id}_gtFine_labellevel3Ids.png
gtFine/{split}/{drive_no}/{img_id}_gtFine_instancelevel3Ids.png
```

### Panoptic Challenge

Furthermore for training, panoptic masks needs to be generated as described below resulting in the following files:
```
gtFine/{split}_panoptic/{drive_no}_{img_id}_gtFine_panopticlevel3Ids.png
gtFine/{split}_panoptic.json
```
### Detection

The structure is slightly similar to Pascal VOC dataset.
- JPEGImages/<capture_category>/<drive sequence>/<>.jpg for images
- Annotations/<capture_category>/<drive sequence>/<>.xml for Annotations

## Labels

See helpers/anue_labels.py

### Generate Label Masks (for training/evaluation) (Semantic/Instance/Panoptic Segmentation)
```bash
python preperation/createLabels.py --datadir $ANUE --id-type $IDTYPE --color [True|False] --instance [True|False] --num-workers $C
```

- ANUE is the path to the AutoNUE dataset
- IDTYPE can be id, csId, csTrainId, level3Id, level2Id, level1Id. 
- color True  generates the color masks
- instance True generates the instance masks with the id given by IDTYPE
- panoptic True generates panoptic masks in the format similar to COCO. See the modified evaluation scripts here: https://github.com/AutoNUE/panopticapi
- C is the number of threads to run in parallel

For the semantic segmentation challenge, masks should be generated using IDTYPE of level3Id and used for training models (similar to trainId in cityscapes). This can be done by the command:
```bash
python preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers $C
```
For the instance segmentation challenge, instance masks should be generated by the following comand:
```bash
python preperation/createLabels.py --datadir $ANUE --id-type id --num-workers $C
```

The generated files:

- _gtFine_labelLevel3Ids.png will be used for semantic segmentation
- _gtFine_instanceids.png will be used for instance segmentation
- _gtFine_panopticLevel3Ids.png will be used for panoptic segmentation under the folder gtFine/{split}_panoptic and the gtFine/{split}_panoptic.json

### Detection

We use subset of labels from helpers/anue_labels.py.

We have person(level3Id: 4 , Trainable : True), rider (level3Id: 5, Trainable : True), car (level3Id: 9, Trainable : True), truck (level3Id: 10, Trainable : True),  bus(level3Id: 11, Trainable : True), motorcycle(level3Id: 6, Trainable : True), bicycle(level3Id: 7, Trainable : True), autorickshaw(level3Id: 8, Trainable : True), animal(level3Id: 4 , Trainable : True), traffic light(level3Id: 18, Trainable : True), traffic sign(level3Id: 19, Trainable : True), vehicle fallback (level3Id: 12, Trainable : False), caravan (level3Id: 12, Trainable : False), trailer (level3Id: 12, Trainable : False), train (level3Id: 12, Trainable : False).

Note : We train based on level3Id’s and only those labels which are mentioned as trainable and report accuracies on them.


## Viewer

First generate label masks as described above. To view the ground truths / prediction masks at different levels of heirarchy use:
```bash
python viewer/viewer.py ---datadir $ANUE
```

- ANUE has the folder path to the dataset or prediction masks with similar file/folder structure as dataset.

TODO: Make the color map more sensible.


## Evaluation

### Semantic Segmentation

First generate labels masks with level3Ids as described before. Then
```bash
python evaluate/evaluate_mIoU.py --gts $GT  --preds $PRED  --num-workers $C
```

- GT is the folder path of ground truths containing <drive_no>/<img_no>_gtFine_labellevel3Ids.png 
- PRED is the folder paths of predictions with the same folder structure and file names.
- C is the number of threads to run in parallel


### Constrained Semantic Segmentation

First generate labels masks with level1Ids as described before. Then
```bash
python evaluate/idd_lite_evaluate_mIoU.py --gts $GT  --preds $PRED  --num-workers $C
```

- GT is the folder path of ground truths containing <drive_no>/<img_no>_gtFine_labellevel1Ids.png 
- PRED is the folder paths of predictions with the same folder structure and file names.
- C is the number of threads to run in parallel


### Instance Segmentation


First generate instance label masks with ID_TYPE=id, as described before. Then
```bash
python evaluate/evaluate_instance_segmentation.py --gts $GT  --preds $PRED 
```

- GT is the folder path of ground truths containing <drive_no>/<img_no>_gtFine_labellevel3Ids.png 
- PRED is the folder paths of predictions with the same folder structure and file names. The format for predictions is the same as the cityscapes dataset. That is a .txt file where each line is of the form "<instance_mask_png> <label id> <conf score>". Note that the ID_TYPE=id is used by this evaluation code.
- C is the number of threads to run in parallel

### Panoptic Segmentation

Please use https://github.com/AutoNUE/panopticapi

### Detection

```bash
python evaluate/evaluate_detection.py --gts $GT  --preds $PRED 
```
- GT is the folder path of ground truths containing Annotations/<capture_category>/<drive sequence>/<>.xml
- PRED is the folder path of predictions with generated outputs in idd_det_<image_set>_<level3Id>.txt format. Here image_set can take {train,val,test}, while level3Id for all trainable labels has to present.



## Acknowledgement

Some of the code was adapted from the cityscapes code at: https://github.com/mcordts/cityscapesScripts/ 
Some of the code was adapted from https://github.com/rbgirshick/py-faster-rcnn
Some of the code was adapted from https://github.com/cocodataset/panopticapi

