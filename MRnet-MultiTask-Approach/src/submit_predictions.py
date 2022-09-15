import torch
import numpy as np
import sys

from models import MRnet


INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

def _resize_image(image):
    """Resize the image to `(3,224,224)` and apply 
    transforms if possible.
    """
    # Resize the image
    pad = int((image.shape[2] - INPUT_DIM)/2)
    image = image[:,pad:-pad,pad:-pad]
    image = (image-np.min(image))/(np.max(image)-np.min(image))*MAX_PIXEL_VAL
    image = (image - MEAN) / STDDEV

    image = np.stack((image,)*3, axis=1)
    
    image = torch.FloatTensor(image)
    return image

def _get_images(axial_path, coron_path, sagit_path):
    axial = np.load(axial_path)
    coron = np.load(coron_path)
    sagit = np.load(sagit_path)
    
    # Load images and add a extra dimension
    axial_tensor = _resize_image(axial).unsqueeze(dim=0)
    coron_tensor = _resize_image(coron).unsqueeze(dim=0)
    sagit_tensor = _resize_image(sagit).unsqueeze(dim=0)
    
    return [axial_tensor, coron_tensor, sagit_tensor]


if __name__ == '__main__':
    input_csv_path = sys.argv[1] # 'valid-paths.csv'
    preds_csv_path = sys.argv[2] # 'pred.csv' # sys.argv[2]

    paths = []
    for i, fpath in enumerate(open(input_csv_path).readlines()):
        if 'axial' in fpath:
            axial_path = fpath.strip()
        elif 'coronal' in fpath:
            coron_path = fpath.strip()
        elif 'sagittal' in fpath:
            sagit_path = fpath.strip()

        if i % 3 == 2:
            paths.append((axial_path, coron_path, sagit_path))
    
    all_predictions = {'abnormal' : [], 'acl' : [], 'meniscus' : []}
    
    # Loading all models
    diseases = ['abnormal','acl','meniscus']
    model_paths = [
        './src/model_final_train_abnormal_val_auc_0.9394_train_auc_0.9379_epoch_30.pth', 
        './src/model_final_train_acl_val_auc_0.9602_train_auc_0.9956_epoch_32.pth', 
        './src/model_final_train_meniscus_val_auc_0.7837_train_auc_0.9161_epoch_30.pth'
    ]

    for model_path, disease in zip(model_paths,diseases):
        
        model = MRnet()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['model_state_dict'])

        if torch.cuda.is_available():
            model = model.cuda()

        for axial_path, coron_path, sagit_path in paths:

            images = _get_images(axial_path, coron_path, sagit_path)

            if torch.cuda.is_available():
                images = [image.cuda() for image in images]

            with torch.no_grad():
                output = model(images)
                prob = torch.sigmoid(output).item()
                all_predictions[disease].append(prob)
    
    # Write all results in .csv file
    with open(preds_csv_path, 'w') as csv_file:        
        for i in range(len(all_predictions['acl'])):
            csv_file.write(','.join(
                [str(all_predictions['abnormal'][i]), str(all_predictions['acl'][i]), str(all_predictions['meniscus'][i])]))
            csv_file.write('\n')
