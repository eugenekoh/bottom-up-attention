
"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# python2 tools/tsv_gen.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out feature/custom.feature.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split custom

import pickle
import csv
import sys
import os
import tqdm 
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'images']

if __name__ == '__main__':
    with open('./data/features/yfcc.tsv') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
        found = set(x['image_id'] for x in reader)

    files = os.listdir('/mnt/vol_b/yfcc_images')
    want = set(f.split('.')[0] for f in files)
    print(len(found))
    print(len(want))

    missing = want - found
    with open('./data/features/missing.pkl', 'w') as f:
        pickle.dump(missing, f)

    print(len(missing)) 