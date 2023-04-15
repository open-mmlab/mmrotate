# Semi-automatic Object Detection Annotation with MMRotate and Label-Studio

Annotation data is a time-consuming and laborious task. This article introduces how to perform semi-automatic annotation using the RTMDet-R algorithm in MMRotate in conjunction with Label-Studio software. Specifically, using RTMDet-R to predict image annotations and then refining the annotations with Label-Studio. Community users can refer to this process and methodology and apply it to other fields.

- RTMDet-R: RTMDet-R is a high-precision single-stage object detection algorithm developed by OpenMMLab, open-sourced in the MMRotate rotated object detection toolbox. Its open-source license is Apache 2.0, and it can be used freely without restrictions by industrial users.

- [Label Studio](https://github.com/heartexlabs/label-studio) is an excellent annotation software covering the functionality of dataset annotation in areas such as image classification, object detection, and segmentation.

In this article, we will use [DOTA](https://captain-whu.github.io/DOTA/index.html) images for semi-automatic annotation.

## Environment Configuration

To begin with, you need to create a virtual environment and then install PyTorch and MMCV. In this article, we will specify the versions of PyTorch and MMCV. Next, you can install MMDetection, MMRotate, Label-Studio, and label-studio-ml-backend using the following steps:

Create a virtual environment:

```shell
conda create -n rtmdet python=3.9 -y
conda activate rtmdet
```

Install PyTorch:

```shell
# Linux and Windows CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Linux and Windows CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# OSX
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
```

Install MMCV:

```shell
pip install -U openmim
mim install "mmcv>=2.0.0rc0"
# Installing mmcv will automatically install mmengine
```

Install MMDetection:

```shell
git clone https://github.com/open-mmlab/mmdetection -b dev-3.x
cd mmdetection
pip install -v -e .
```

Install MMRotate:

```shell
git clone https://github.com/open-mmlab/mmrotate/ -b dev-1.x
cd mmrotate
pip install -v -e .
```

Install Label-Studio and label-studio-ml-backend:

```shell
# Installing Label-Studio may take some time, if the version is not found, please use the official source
pip install label-studio==1.7.2
pip install label-studio-ml==1.0.9
```

Download the RTMDet-R weights:

```shell
cd path/to/mmrotate
mkdir work_dirs
cd work_dirs
wget https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota/rotated_rtmdet_m-3x-dota-beeadda6.pth
```

## Start the Service

Start the RTMDet-R backend inference service:

```shell
cd path/to/mmrotate
label-studio-ml start projects/LabelStudio/backend_template --with \
config_file=configs/rotated_rtmdet/rotated_rtmdet_m-3x-dota.py \
checkpoint_file=./work_dirs/rotated_rtmdet_m-3x-dota-beeadda6.pth \
device=cpu \
--port 8003
# Set device=cpu to use CPU inference, and replace cpu with cuda:0 to use GPU inference.
```

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/label_studio_ml_start.png)

The RTMDet-R backend inference service has now been started. To configure it in the Label-Studio web system, use http://localhost:8003 as the backend inference service.

Now, start the Label-Studio web service:

```shell
label-studio start
```

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/label_studio_start.png)

Open your web browser and go to http://localhost:8080/ to see the Label-Studio interface.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/sign_up.png)

Register a user and then create an RTMDet-R-Semiautomatic-Label project.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/create_project.png)

Download the example DOTA images by following [MMRotate Preparing DOTA Dataset](https://github.com/open-mmlab/mmrotate/blob/main/tools/data/dota/README.md) and import them using the Data Import button.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/data_import.png)

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/data_import_from_file.png)

Then, select the Object Detection With Bounding Boxes template.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/templete_select.png)

```shell
plane
ship
storage-tank
baseball-diamond
tennis-court
basketball-court
ground-track-field
harbor
bridge
large-vehicle
small-vehicle
helicopter
roundabout
soccer-ball-field
swimming-pool
```

Then, copy and add the above categories to Label-Studio and click Save.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/add_label.png)

In the Settings, click Add Model to add the RTMDet-R backend inference service.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/add_model.png)

Click Validate and Save, and then click Start Labeling.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/validate_and_save.png)

If you see Connected as shown below, the backend inference service has been successfully added.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/connected.png)

## Start Semi-Automatic Labeling

Click on Label to start labeling.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/click_label.png)

We can see that the RTMDet-R backend inference service has successfully returned the predicted results and displayed them on the image. However, we noticed that there are some missed predicted bounding boxes.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/label_result.png)

We manually add some annotation boxes, adjust the position of the boxes to get the following corrected annotation, and then click Submit to complete the annotation of this image.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/label_result_refined.png)

After submitting all images, click export to export the labeled dataset in COCO format.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/label_export.png)

Use VSCode to open the unzipped folder to see the labeled dataset, which includes the images and the annotation files in JSON format.

![](https://github.com/fengshiwest/mmrotate/blob/add_image/projects/LabelStudio/images/json_show.png)

At this point, the semi-automatic labeling is complete. We can use this dataset to train a more accurate model in MMRotate and then continue semi-automatic labeling on newly collected images with this model. This way, we can iteratively expand the high-quality dataset and improve the accuracy of the model.
