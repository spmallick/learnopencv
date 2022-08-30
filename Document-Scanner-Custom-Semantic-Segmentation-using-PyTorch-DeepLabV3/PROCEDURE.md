### Procedure for Generating Synthetic Dataset and Training

1. Create two folders: `DOCUMENTS\datasets` and `DOCUMENTS\CHOSEN`.

2. Download the documents from the sources mentioned in the post. Note that there might be some extra straightforward steps you might to perform for downloading. The *SmartDoc QA* set is of size 13+ GB, and we have used only 85 images from the entire set; you can skip it if you want.

3. After downloading, create a folder for each set in the `DOCUMENTS\datasets` directory and copy document images from each set into appropriate folders. Eg. For *DocVQA*, you will have two folders, `val\documents` and `test\documents` which contain the document images. Copy all images from both folders into `DOCUMENTS\datasets\docvqa_images` folder.

4. Run the `generate_doc_set.py` script (with correct folder paths), which should randomly copy documents and create the mask for the same in the `DOCUMENTS\CHOSEN\images` and `DOCUMENTS\CHOSEN\masks` folders, respectively.

5. We have resized the document and mask images (while maintaining the image's aspect ratio) to have a max dimension of size `640`.

```python
# Resize command
python resize.py -s DOCUMENTS\CHOSEN\images -d DOCUMENTS\CHOSEN\resized_images -x 640
python resize.py -s DOCUMENTS\CHOSEN\masks -d DOCUMENTS\CHOSEN\resized_masks -x 640
```

6. Your final directory structure should then look something like this:

```HTML
├───DOCUMENTS
│   ├───CHOSEN
│   │   ├───images        
│   │   ├───masks
│   │   ├───resized_images
│   │   └───resized_masks 
│   └───datasets
│       ├───annotated_640
│       ├───docvqa_images
│       ├───formsE-H_images
│       ├───FUNSD_images
│       ├───kaggle_noisy_images
│       └───nouvel_images
```

7. For downloading background images, first clone the [fork of Google Images Download repository](https://github.com/Joeclinton1/google-images-download) and copy the `download_google_images.py` script inside the repository and execute it (from within the repository). This will start downloading Google search images for various `search_queries` mentioned in the script. It'll first create a separate folder for query within the `downloads` folders (inside the repository folder) and place all the downloaded images in that folder.

8. Create a root folder `background_images` that contains all the downloaded images. First, convert all images from different formats to *.jpg*, then remove duplicate images. You can perform this task either manually or write a script. The script aims to compare the content of one image (anchor) with the rest of the images and then discard the duplicate ones if they are similar.

9. At this point you should have folders:- `DOCUMENTS\CHOSEN\resized_*` and `background_images`.

10. Now, execute the `create_dataset.py` script (by setting the path appropriately in the script), which should generate the synthetic image and mask dataset in folders `final_set\images` and `final_set\masks`, respectively. We have kept the `document: background` ratio as `1:6`; you can change it as you wish.  

11. To split the datasets into train and valid sets, execute the `split_dataset.py` script (by setting the path appropriately and the `img_per_doc` variable as in the previous step within the script ). The script also resizes (maintaining aspect ratio) the images to have a max dimension of 480. You can choose to skip it by changing the code accordingly. The code for simply copying the original image and mask is currently commented out.

12. You are now ready to train your custom semantic segmentation model. Checkout the `Custom_training_deeplabv3_mbv3.ipynb` notebook for training the DeeplabV3 with MobilenetV3-Large backbone. Set the number of `epochs` appropriately.

#### Dataset and Trained Model Download Links

1. Resized final dataset: [drive](https://drive.google.com/file/d/1rRmNRQgSW3k09B76AAi5LfGztRSDOXVf/view?usp=sharing)
2. Model - MobileNetV3-Large backend: [drive](https://drive.google.com/file/d/1ROtCvke02aFT6wnK-DTAIKP5-8ppXE2a/view?usp=sharing)
3. Model - Resnet50 backend: [drive](https://drive.google.com/file/d/1DEl6qLckFChSDlT_oLUbO2JpN776Qx-g/view?usp=sharing)
