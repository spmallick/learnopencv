from argparse import ArgumentParser
from PIL import Image
import os
import glob
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys
import cPickle

sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )

from anue_labels import *
from helper_eval_detection import idd_eval

_classes_level3Ids = (
  			4, #("person", "animal"),
	   		5, #"rider",
  			6,#"motorcycle",
	        7,#"bicycle",
	        8,#"autorickshaw",
	    	9,#"car",
	     	10,#"truck",
			11,#"bus"
			18,#"traffic light" ,
 			19,#"traffic sign",'''
 			)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gts', default="")
    parser.add_argument('--preds', default="")
    parser.add_argument('--image-set', default="test")
    parser.add_argument('--output-dir', default="output")
    
    args = parser.parse_args()

    return args

def _get_idd_results_file_template():
    # idd_det_test_<level3Id>.txt
    filename = 'idd_det_' + args.image_set + '_{:s}.txt'
    path = os.path.join(
        args.preds,
        filename)
    return path

def _do_eval(output_dir = 'output'):
    annopath = os.path.join(
        args.gts,
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        args.gts,
        args.image_set +'.txt')
    cachedir = os.path.join(args.gts, 'annotations_cache')
    aps = []
    # This metric is similar to PASCAL VOC metric in 2010
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls_id in enumerate(_classes_level3Ids):
        filename = _get_idd_results_file_template().format(str(cls_id))
        rec, prec, ap = idd_eval(
            filename, annopath, imagesetfile, cls_id, cachedir, ovthresh=0.5)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls_id, ap))
        with open(os.path.join(output_dir, str(cls_id) + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')

def main(args):
	_do_eval(args.output_dir)
	return

if __name__ == '__main__':
    args = get_args()
    main(args)



