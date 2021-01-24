#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# python2 tools/tsv_gen.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out feature/custom.feature.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split custom

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import pickle
import caffe
import argparse
import time
import os
import sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import os
import pprint
import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'images']
LABELNAMES = ['image_id', 'list']
data_path = 'data/genome'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 100

def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    if im is None:
        os.remove(im_file)
        return 
    #print(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    boxes = cls_boxes[keep_boxes]
    image_width = np.size(im, 1)
    image_height = np.size(im, 0)
    features = pool5[keep_boxes]

    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    scaled_width = box_width / image_width
    scaled_height = box_height / image_height
    scaled_x = boxes[:, 0] / image_width
    scaled_y = boxes[:, 1] / image_height
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]
    spatial_features = np.concatenate(
         (scaled_x,
          scaled_y,
          scaled_x + scaled_width,
          scaled_y + scaled_height,
          scaled_width,
          scaled_height),
         axis=1)
    full_features = np.concatenate((features, spatial_features), axis=1)
    fea_base64 = base64.b64encode(full_features).decode('utf-8')
    fea_info = {'features': fea_base64, 'num_boxes': boxes.shape[0]}


    objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    labels = []

    for i in range(len(keep_boxes)):
        labels.append({
            'class': classes[objects[i]+1],
            'rect' : cls_boxes[keep_boxes][i].tolist(),
            'conf' : max_conf[keep_boxes][i]
        })

    return ({'image_id' : image_id, 'images' : fea_info }, {'image_id' : image_id,'list': labels}  )


def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default='resnet101_faster_rcnn_final.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default="experiments/cfgs/faster_rcnn_end2end_resnet.yml", type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--errors', dest='errors_file')
    parser.add_argument('--skip', dest='skip_file')
    parser.add_argument('--images', dest='image_path')
    parser.add_argument('--features', dest='features_file')
    parser.add_argument('--labels', dest='labels_file')

    args = parser.parse_args()
    return args


def generate_tsv(gpu_id, prototxt, weights, image_ids, features_file, labels_file, errors_file, skip_file):
    # First check if file exists, and if it is complete

    with open(skip_file) as f:
        wanted = set(pickle.load(f))

    found_ids = set()
    if os.path.exists(features_file):
        with open(features_file) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            found_ids = set(x['image_id'] for x in reader)

    errors = set()
    if os.path.exists(errors_file):
        with open(errors_file) as csvfile:
            reader = csv.reader(csvfile)
            errors = set(x[0] for x in reader)

    missing = wanted - found_ids - errors
    
    if len(missing) == 0:
        print 'GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids))
    else:
        print 'GPU {:d}: missing {:d}/{:d}, errors:{:d}'.format(gpu_id, len(missing), len(wanted), len(errors))

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    missing_items = [x for x in image_ids if x[1] in missing]
    
    with open(features_file, 'ab') as tsvfile, open(labels_file, 'ab') as labeltsvfile, open(errors_file, 'a') as csvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
        Labelwriter = csv.DictWriter(labeltsvfile, delimiter = '\t',fieldnames = LABELNAMES)
        error_writer = csv.writer(csvfile)

        for im_file,image_id in tqdm.tqdm(missing_items):
            try:
                X = get_detections_from_im(net, im_file, image_id)
            except:
                error = 'exception error for {}'.format(im_file)
                tqdm.tqdm.write(error)
                error_writer.writerow([image_id, error])
                continue
            if X is None:
                error = 'empty image for {}'.format(im_file)
                tqdm.tqdm.write(error)
                error_writer.writerow([image_id, error])
                continue

            writer.writerow(X[0])
            Labelwriter.writerow(X[1])

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    image_ids = []
    files = os.listdir(args.image_path)
    for f in files:
        image_id = f.split('.')[0]
        filepath = os.path.join(args.image_path, f)
        image_ids.append((filepath, image_id))

    random.shuffle(image_ids)
    # Split image ids between gpus
    image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]

    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for i,gpu_id in enumerate(gpus):
        features_file = '%s.%d' % (args.features_file, gpu_id)
        labels_file = '%s.%d' % (args.labels_file, gpu_id)
        p = Process(target=generate_tsv,
                    args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], features_file, labels_file, args.errors_file, args.skip_file))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
