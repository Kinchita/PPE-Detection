
# PPE-Detection

A brief description of what this project does and who it's for


## Overview
This repository contais the code for detection of Personal Protection Equipment(PPE) on images, the datase is sourced from Kaggle. It can identify ten different items such as hard hats, vests, gloves, masks etc. I have used You Look Once Only(YOLO) algorithm for this detection. The motivation behind this project is safety of construction workers.
 
## Introduction
Accidents on construction sites have increased three fold
since 2021in Mumbai, according to a study there are 38 fatal
accidents everyday across construction sites in India. These
records can be shocking, but our workers have been risking
their lives every day. Their safety is of utmost importance in
today’s time and date, developing a computer vision model
using the technique of YOLO will provide a different way
of industrial surveillance to contractors. The reason for such
accidents is unsafe culture on construction sites, which leads to
mishaps like these more such developments have happened in
the field of computer vision which has helped in controlling or
minimising these accidents.With the help of a robust YOLO
model one can detect multiple safety gear items simultaneously
in the same frame
1) Hard hats and Safety Vests are of utmost importance
on construction sites. These hard hats save a lot of
workers from life endangering situations by providing
solid support to their heads.
2) Identifying and detecting safety gear items such as
Hard hats, Safety vests, Gloves etc is the goal .
3) It is often challenging to simultaneously detect so many
items together in one frame but YOLO provides a better
technique to do this detection.
4) The aim is to improve the identification and detection of
safety gear items using Computer Vision


## Data Source 
The subsequent stages for project involves downloading
and organizing one dataset, ”Construction Site Safety Image
Dataset by Roboflow” sourced from Kaggle. The script unzips
the dataset and organizes them into separate directories,
simplifying data management and setting the stage for data
processing. Here because we the datset is by Roboflow it
eases up Data Annotation, Data Augmentation as well as Data
Management.
## Installation
---> Downloading the Dataset

Using Google Colab, we first install Kaggle so we can interact with the interface of Kaggle smoothly 

```bash
  !pip install kaggle
```
The next command line will help in creating a hidden directory. This is used for the API key which we wil be using ahead.    

```bash 
!mkdir ~/.kaggle
```
This command line further allows the contents of the kaggle.json file which we have attatined as an API token from Kaggle. These files will be copied from the current directory to the directory we created above 'kaggle'

```bash
!cp kaggle.json ~/.kaggle/kaggle.json
```

The next command that sets the permissions of the kaggle.json file in the .kaggle directory to read and write 

```bash
!chmod 600 ~/.kaggle/kaggle.json
```

To download to the current directory in the filesystem we will usethe following command line 

```bash 
!kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow
```

Now we extract and place the files in the cureent directory

```bash
!unzip /content/construction-site-safety-image-dataset-roboflow.zip
```
---> Cloning the YOLOv9 

Cloning YOLOv9 from github repository

```bash 
!git clone https://github.com/SkalskiP/yolov9.git
```

---> Configuration 

```bash 

dataDir = '/content/css-data/'
workingDir = '/content/'
```
Here, I have created two directories. The first one allows storing the CSS(Casscadig Sheet Style) files. Also, the second directory represents the working where the code or scripts are executed 

```bash
num_classes = 10
classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machine','vehicle']
```

Using these two variables to specify the number of classes and the labels associated with each class. This will help in easier reference and understanding models output 

```bash 
import yaml
import os

file_dict = {
    'train': os.path.join(dataDir, 'train'),
    'val': os.path.join(dataDir, 'valid'),
    'test': os.path.join(dataDir, 'test'),
    'nc': num_classes,
    'names': classes
}

with open(os.path.join(workingDir,'yolov9', 'data.yaml'), 'w+') as f:
  yaml.dump(file_dict, f)
```

Here, we are first importing the necesaary modules viz. YAML module, which provides functions for reading and writing YAML files and os  which provides functions for interacting with the operating system, such as file handling and directory manipulation. 

Then we construct a dict which consists of all the keys such as train, val, test, which repectively represent the directories to the training, validation and testing dataset. The other keys viz. nc and names represent the number of classes and list of labels respectively.

---> Dowloading YOLOv9 weights
 
```bash 
!wget  https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
```

Downloading pretrained weights will make the training easier for thr models


## Training 

```bash 
cd yolov9
```
This is doneto make sure we're working in the correct directory and it won't be changes unless we change it. 

```bash 
!pip install -r requirements.txt -q
```

To install all the required packages for training this model 

```bash
!python train_dual.py --workers 8 --batch 4  --img 640 --epochs 50 --data /content/yolov9/data.yaml --weights /content/yolov9-e.pt --device 0 --cfg /content/yolov9/models/detect/yolov9.yaml --hyp /content/yolov9/data/hyps/hyp.scratch-high.yaml
```

By giving the arguments we are specifying certain things such as 640 pixels of images, 4 batches, 50 epochs foe a better ouput along with this also the path to the correct files for cong=figuration, weights and data is specified. This command lne allows to initiate training of a dual-objective model with specific parameters.


## Inferences 
```bash 
!python detect.py --img 640 --conf 0.1 --device 0 --weights /content/yolov9/runs/train/exp/weights/best.pt --source /content/css-data/test/images/002551_jpg.rf.ce4b9f934161faa72c80dc6898d37b2d.jpg
```

Once the model is trained we can, start the testing which is shown in the code above given the specific parameters.

```bash
from IPython.display import Image
Image(filename="/content/yolov9/runs/detect/exp4/002551_jpg.rf.ce4b9f934161faa72c80dc6898d37b2d.jpg", width=600)
```

The path for the trained image is then used to show the output of this model 
## Output 

We can see the output of this now trained model here in this folder 

https://drive.google.com/drive/folders/1LaM_zU8aruse1MoHp0B7RGdw56yYpKZ_