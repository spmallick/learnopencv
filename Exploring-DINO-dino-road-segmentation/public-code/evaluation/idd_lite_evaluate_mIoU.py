from argparse import ArgumentParser
from PIL import Image
import os
import glob
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys

res = 128

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gts', default="/ssd_scratch/cvit/girish.varma/dataset/anue_test/gtFine/test")
    parser.add_argument('--preds', default="")
    parser.add_argument('--prefix', default="_label.png")
    parser.add_argument('--res', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=10)
    
    args = parser.parse_args()

    return args


def add_to_confusion_matrix(gt, pred, mat):
#    print(pred.shape)
#    print(pred.size[0],pred.size[1])

    if (pred.shape[0] != gt.shape[0]):
        print("Image widths of " + pred + " and " + gt + " are not equal.")
    if (pred.shape[1] != gt.shape[1]):
        print("Image heights of " + pred + " and " + gt + " are not equal.")
    if ( len(pred.shape) != 2 ):
        print("Predicted image has multiple channels.")
    W  = pred.shape[0]
    H = pred.shape[1]
    P = H*W

    pred = pred.flatten()
    gt = gt.flatten()


    for (gtp,predp) in zip(gt, pred):
        if gtp == 255 or gtp >7 :
            gtp = 7
        if predp == 255 or predp >7:
            predp = 7
        mat[gtp, predp] += 1

    return mat

def higher_level_mat(mat, mapping):
    n = len(mapping)
    h_mat = np.zeros((n,n), dtype=np.ulonglong)

    for x_id in range(n):
        for y_id in range(n):
            x_h_ids = mapping[x_id]
            y_h_ids = mapping[y_id]
            for x_h_id in x_h_ids:
                for y_h_id in y_h_ids:
                    h_mat[x_id,y_id] += mat[x_h_id, y_h_id]

    return h_mat


def generalized_eval_ious(mat):
    n = mat.shape[0]
    ious = np.zeros(n)
    for l in range(n):
        tp = np.longlong(mat[l,l])
        fn = np.longlong(mat[l,:].sum()) - tp

        notIgnored = [i for i in range(n) if not i==l]
        fp = np.longlong(mat[notIgnored,l].sum())
        denom = (tp + fp + fn)
        if denom == 0:
            print('error: denom is 0')

        ious[l] =  float(tp) / denom
    return ious


def print_scores(ious, names, heading):
    print('---------------------------------------------')
    print(heading)
    print('---------------------------------------------')
    for (iou, name) in zip(ious, names):
        print(f'{name}\t\t:{iou}')
    print('---------------------------------------------')
    print(f'mIoU\t\t:{ious.mean()}')
    print('---------------------------------------------')


    



def ious_at_all_levels(mat):
    global res


    l1_mat = mat
    l1_ious = generalized_eval_ious(l1_mat)
    
    np.save(f'eval_results/lite_ious_l1_{res}', np.array(l1_ious))
    
    np.save(f'eval_results/lite_cm_l1_{res}', np.array(l1_mat))
    
    l1_names = ['drivable', 'non-drivable', 'living-things', 'vehicles', 'road-side-objs', 'far-objects', 'sky']


    print_scores(l1_ious, l1_names, "Level 1 Scores")

    

def process_pred_gt_pair(pair):
    global res
    W,H = 1920, 1080
    if res == 720:
        W,H = 1280, 720
        # print(W,H)
    if res == 480:
        W,H = 858, 480
    if res == 240:
        W,H = 426, 240
    if res == 128:
        W,H = 256, 128
    
    

    pred, gt = pair
    # tqdm.tqdm.write(pred, gt)
    confusion_matrix = np.zeros(shape=(7+1, 7+1),dtype=np.ulonglong)
    try:
        gt = Image.open(gt)
        # print(gt.size)
        if gt.size != (W, H):
            gt = gt.resize((W, H), resample=Image.NEAREST)
        gt  = np.array(gt)
    except:
        print("Unable to load " + gt)
        os._exit(1)

    try:
        pred = Image.open(pred)
        if pred.size != (W, H):
            pred = pred.resize((W, H), resample=Image.NEAREST)
        pred = np.array(pred)
    except:
        print("Unable to load " + pred)
        os._exit(1)

    # plt.matshow(gt)
    # plt.show()
    # plt.matshow(pred)
    # plt.show()


    
    

    # print(pred.size,gt.size)
    
    add_to_confusion_matrix(gt, pred, confusion_matrix)

    return confusion_matrix

import tqdm

class_names = ['road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky', 'misc']



def eval_ious(mat):
    ious = np.zeros(7+1)
    for l in range(7):
        tp = np.longlong(mat[l,l])
        fn = np.longlong(mat[l,:].sum()) - tp

        notIgnored = [i for i in range(7) if not i==l]
        fp = np.longlong(mat[notIgnored,l].sum())
        denom = (tp + fp + fn)
        if denom == 0:
            print('error: denom is 0')

        ious[l] =  float(tp) / denom

    return ious[:-1]


def main(args):
    global res
    res = args.res
    confusion_matrix    = np.zeros(shape=(7+1, 7+1),dtype=np.ulonglong)
    gts_folders         =  glob.glob(args.gts + '/*')
    pred_folders        = [ gtf.replace(args.gts, args.preds) for gtf in gts_folders ]
    # print(gts_folders)

    gts     = []
    preds   = []
    for i, gtf in enumerate(gts_folders):
        
        g = glob.glob(gtf+'/*_label.png')
        g = [ lab for lab in g if not lab.endswith("_inst_label.png") ]
        
        # print(g)
        p = [ j.replace(gtf, pred_folders[i]) for j in g] #.replace('_gtFine_labellevel3Ids.png','_leftImg8bit.png')
        gts += g
        preds += p

    # print(len(gts))

    pairs = [(preds[i], gts[i]) for i in range(len(gts))]

    pool = Pool(args.num_workers)

    results = list(tqdm.tqdm(pool.imap(process_pred_gt_pair, pairs), total=len(pairs)))
    pool.close()
    pool.join()

    for i in range(len(results)):
        confusion_matrix += results[i]

    os.makedirs('eval_results', exist_ok=True)
    ious_at_all_levels(confusion_matrix)
    
    np.save(f'eval_results/lite_cm_{res}',confusion_matrix)

    ious = eval_ious(confusion_matrix)
    np.save(f'eval_results/lite_ious_{res}', np.array(ious))

    # for i in range(26):
    #     print(f'{class_names[i]}:\t\t\t\t {ious[i]*100}')
    print(ious)
    print(f'mIoU:\t\t\t\t{ious.mean()*100}')

        
        
if __name__ == '__main__':
    args = get_args()
    main(args)
