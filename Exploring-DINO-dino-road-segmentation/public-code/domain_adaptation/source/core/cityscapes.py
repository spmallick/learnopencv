from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument("datadir",help="path to dataset")
parser.add_argument("savedir",help="path to save directory")

###############test################
#dd = '/raid/datasets/SemanticSegmentation/domain_adaptation/Cityscapes'
###################################


def getImg(lbl):
    '''
    returns corresponding image to a label, specific to Cityscapes
    '''
    osfx = 'gtFine_labelIds.png'
    nsfx = 'leftImg8bit.png'
    img = Path(str(lbl).replace(osfx,nsfx).replace('gtFine','leftImg8bit'))
    return img

def prepCityscapes(dd,sd):
    dd = Path(dd)
    sd = Path(sd)
    labeldirs = [dd/f'gtFine/{lbls}' for lbls in ['train','val']]

    imgs = sd/'Cityscapes/images'
    lbls = sd/'Cityscapes/labels'
    #lbls.mkdir(exist_ok=True)
    #imgs.mkdir(exist_ok=True)
    if not os.path.exists(lbls):
        os.makedirs(lbls)
    if not os.path.exists(imgs):
        os.makedirs(imgs)

    labels = []
    for ld in labeldirs:
        labels+=list(ld.rglob('*gtFine_labelIds.png'))

    for lbl in tqdm(labels):
        img = getImg(lbl)
        assert img.exists() , 'invalid files picked up, aborting'
        shutil.copy(lbl,lbls/f'{lbl.name}')
        shutil.copy(img,imgs/f'{img.name}')


if __name__ == "__main__":
    args = parser.parse_args()
    dd = Path(args.datadir)
    sd = Path(args.savedir)
    print(f'collecting Cityscapes from {dd} into {sd}')
    prepCityscapes(dd,sd)
