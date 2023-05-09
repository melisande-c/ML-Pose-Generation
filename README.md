# ML-Pose-Generation

A repository for multiple ML experiments involving pose detection and generation using the [PennAction](http://dreamdragon.github.io/PennAction/) dataset. The PennAction data set has multiple clips with each frame saved as a jpeg and arrays of keypoint coordinates for the head, shoulders, elbows, hands, hips, knees and feet.

PennAction Reference: *Weiyu Zhang, Menglong Zhu and Konstantinos Derpanis,  "From
Actemes to Action: A Strongly-supervised Representation for
Detailed Action Understanding" International Conference on
Computer Vision (ICCV). Dec 2013.*

Currently, the two experiments use a modified version of the pretrained VGG architecture from the paper [Very Deep Converlutional Networks for Large Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf).

VGG Reference: *Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.*

## Running on Google Colab

```
# Download the dataset
!wget "https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz"

# Unzip the dataset
!tar -xf "Penn_Action.tar.gz"

# Clone the repository
!git clone "https://github.com/melisande-c/ML-Pose-Generation.git"

# Run the chosen experiment with the desired settings
!python "/content/ML-Pose-Generation/experiments/landmark_regression_detections.py" \
  -dsp "/content/Penn_Action" \
  -sd "/content/Outputs/landmark_regression_detection" \
  -e 10 \
  -bs 64 \
  -tf 0.5
```

## Repository

* [src](src): Code base for models, datasets and other various utilities.
* [experiments](experiments): Training scripts for various experiments.

The src code includes a generic [model trainer](src/utils/ModelTrainer.py) to ensure consistent training and logging of training metadata between experiments; a custom [dataset class](src/data/datasets.py) for loading the PennAction dataset; and [classes](src/models/VGGMod.py) to adapt VGG to these specific use cases.
