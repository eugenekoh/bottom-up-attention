#!/bin/bash
while true
do
python tools/tsv_gen.py \
--gpu "0,1,2" \
--cfg cfgs/faster_rcnn_end2end_resnet.yml \
--def model/test.prototxt \
--net resnet101_faster_rcnn_final.caffemodel \
--errors data/errors.csv \
--skip vast_allocation.pkl \
--images ../yfcc_images \
--features data/features/yfcc.tsv \
--labels data/features/labels.yfcc.tsv
sleep 10
done
