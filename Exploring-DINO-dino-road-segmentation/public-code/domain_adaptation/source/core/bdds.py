from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
import shutil
import os

#print("here")
parser = argparse.ArgumentParser()
parser.add_argument("datadir",help="path to dataset")
parser.add_argument("savedir",help="path to save directory")
#print(parser.parse_args())
### test ###
#dd = Path('/raid/datasets/SemanticSegmentation/domain_adaptation/bdds/')
#lbl = '/raid/datasets/SemanticSegmentation/domain_adaptation/bdds/labels/train/0004a4c0-d4dff0ad_train_id.png'
#
############/home/cvit/rohit/autoneu/github/public-code/domain_adaptation/source/core

def getImg(lbl,dd):
    '''
    returns corresponding image to a label, specific to bdd
    '''
    osfx = '_train_id.png'
    nsfx = '.jpg'
    return dd/f'images/train/{lbl.name.replace(osfx,nsfx)}'

def prepBDD(dd,sd):
    assert dd.exists() , f'dataset directory doesn\'t exist'
    d_strat = pd.read_csv('./domain_adaptation/source/core/csvs/stratified_bdds.csv',header=None)
    strp = '/raid/datasets/SemanticSegmentation/domain_adaptation/bdds/'

    lbls = sd/'BDD/labels'
    imgs = sd/'BDD/images'
    #lbls.mkdir(exist_ok=True)
    #imgs.mkdir(exist_ok=True)
    if not os.path.exists(lbls):
        os.makedirs(lbls)
    if not os.path.exists(imgs):
        os.makedirs(imgs)

    for lbl in tqdm(list(d_strat[0])):
        #print("dd",dd)
        #if str(dd)[-1] != "/": dd = str(dd) + "/"
        if str(dd)[-1] != "/": lbl = Path(lbl.replace(strp,str(dd)+"/"))
        else: lbl = Path(lbl.replace(strp,str(dd)))
        img = getImg(lbl,dd)
        #assert img.exists() and lbl.exists() , 'invalid files picked up'
        shutil.copy(lbl,lbls/f'{lbl.name}')
        shutil.copy(img,imgs/f'{img.name}')


if __name__ == "__main__": 
    #print("here1")
    args = parser.parse_args()
    dd = Path(args.datadir)
    sd = Path(args.savedir)
    #print(f'collecting BDDS from {dd} into {sd}')
    prepBDD(dd,sd)


