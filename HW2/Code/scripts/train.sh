#! /bin/bash

python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 1 --fp16 -o -c pretrained_weights/yolox_s.pth 
