# YOLO-v2: Real-Time Object Detection

Still under development. 71 mAP(darknet) and 74mAP(resnet50) on VOC2007 achieved so far.

This is a pre-released version.

### Disclaimer
This is a re-implementation of original yolo v2 which is based on [darknet](https://github.com/pjreddie/darknet).
The arXiv paper is available [here](https://arxiv.org/pdf/1612.08242.pdf).

### Demo

![demo1](https://user-images.githubusercontent.com/3307514/28980832-29bb0262-7904-11e7-83e3-a5fec65e0c70.png)

### Getting started
- Build from source, this is required because this example is not merged, some
custom operators are not presented in official MXNet. [Instructions](http://mxnet.io/get_started/install.html)
- Install required packages: `cv2`, `matplotlib`

### Try the demo
- Download the pretrained [model](https://github.com/zhreshold/mxnet-yolo/releases/download/0.1-alpha/yolo2_darknet19_416_pascalvoc0712_trainval.zip)(darknet as backbone), or this [model](https://github.com/zhreshold/mxnet-yolo/releases/download/v0.2.0/yolo2_resnet50_voc0712_trainval.tar.gz)(resnet50 as backbone) and extract to `model/` directory.
- Run
```
# cd /path/to/mxnet-yolo
python demo.py --cpu
# available options
python demo.py -h
```

### Train the model
- Grab a pretrained model, e.g. [`darknet19`](https://github.com/zhreshold/mxnet-yolo/releases/download/0.1-alpha/darknet19_416_ILSVRC2012.zip)
- (optional) Grab a pretrained resnet50 model, [`resnet-50-0000.params`](http://data.dmlc.ml/models/imagenet/resnet/50-layers/resnet-50-0000.params),[`resnet-50-symbol.json`](http://data.dmlc.ml/models/imagenet/resnet/50-layers/resnet-50-symbol.json), this will produce slightly better mAP than `darknet` in my experiments.
- Download PASCAL VOC dataset.
```
cd /path/to/where_you_store_datasets/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
ln -s /path/to/VOCdevkit /path/to/mxnet-yolo/data/VOCdevkit
```
- Create packed binary file for faster training
```
# cd /path/to/mxnet-ssd
bash tools/prepare_pascal.sh
# or if you are using windows
python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target ./data/train.lst
python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target ./data/val.lst --shuffle False
```
- Start training
```
python train.py --gpus 0,1,2,3 --epoch 0
# choose different networks, such as resnet50_yolo
python train.py --gpus 0,1,2,3 --network resnet50_yolo --data-shape 416 --pretrained model/resnet-50 --epoch 0
# see advanced arguments for training
python train.py -h
```
