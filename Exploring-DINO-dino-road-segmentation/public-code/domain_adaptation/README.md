# AutoNUE2021_domain_adaptation

This repository contains the datasets for Domain Adaptation challenge for AutoNEU 2021, CVPR Workshop. For more details, please visit https://cvit.iiit.ac.in/autonue2021/challenge.html


# Source datasets:
For the Cityscapes dataset, participants are requested to use the fine images (~3200 training samples). Refer: https://www.cityscapes-dataset.com/examples/#fine-annotations. For the other datasets (BDD, GTA and Mapillary), the list of image names are given in the csv files in the folder "Source".

Participants are requested to download the datasets from original websites, given below for easy reference:-
1. https://www.mapillary.com/dataset/vistas?pKey=q0GhQpk20wJm1ba1mfwJmw
2. https://bdd-data.berkeley.edu/ (you might have to click on Advanced tab, and then click on "proceed to bdd-data.berkeley.edu")
3. https://download.visinf.tu-darmstadt.de/data/from_games/

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


Run the following commands **from public-code**:

```
pip3 install requirements.txt
chmod +x domain_adaptation/source/prep_all.sh
./domain_adaptation/source/prep_all.sh
```

This will create a folder "domain_adaptation/source/source_datasets_dir/" where you will find the images and annotations for the source dataset to be used for this competetion.

# Target datasets:

Following commands are updated for the target labels of challenges other than supervised domain adaptation and semantic segmentation, **run them from public-code**:

```
python3 preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers 4 --semisup_da True
python3 preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers 4 --weaksup_da True
python3 preperation/createLabels.py --datadir $ANUE --id-type level3Id --num-workers 4 --unsup_da True
```

The bounding box labels for weakly supervised domain adapation can be downloaded from here: https://github.com/AutoNUE/public-code/tree/master/domain_adaptation/target/weakly-supervised
