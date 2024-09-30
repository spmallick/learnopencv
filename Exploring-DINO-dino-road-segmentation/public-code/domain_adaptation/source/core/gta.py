from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument("datadir",help="path to dataset")
parser.add_argument("savedir",help="path to save directory")

### test ###
dd = Path('/raid/datasets/SemanticSegmentation/domain_adaptation/GTA5/')
lbl = '/raid/datasets/SemanticSegmentation/domain_adaptation/GTA5/labels/labels/00005.png'

############

def getImg(lbl,dd):
    '''
    returns corresponding image to a label, specific to gta
    '''
    return dd/f'images/{lbl.name}'

def prepGTA(dd,sd):
    assert dd.exists() , f'dataset directory doesn\'t exist'
    d_strat = pd.read_csv('./domain_adaptation/source/core/csvs/stratified_GTA.csv',header=None)
    strp = '/raid/datasets/SemanticSegmentation/domain_adaptation/GTA5/labels'

    lbls = sd/'GTA/labels'
    imgs = sd/'GTA/images'
    #lbls.mkdir(exist_ok=True)
    #imgs.mkdir(exist_ok=True)
    if not os.path.exists(lbls):
        os.makedirs(lbls)
    if not os.path.exists(imgs):
        os.makedirs(imgs)

    for lbl in tqdm(list(d_strat[0])):
        if str(dd)[-1] != "/": lbl = Path(lbl.replace(strp,str(dd)+"/"))
        else: lbl = Path(lbl.replace(strp,str(dd)))
        img = getImg(lbl,dd)
        assert img.exists() and lbl.exists() , 'invalid files picked up'
        shutil.copy(lbl,lbls/f'{lbl.name}')
        shutil.copy(img,imgs/f'{img.name}')


if __name__ == "__main__": 
    args = parser.parse_args()
    dd = Path(args.datadir)
    sd = Path(args.savedir)
    print(f'collecting GTA from {dd} into {sd}')
    prepGTA(dd,sd)


