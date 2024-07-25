# HPE-HRNet-YOLO
Human Pose Estimation with HRNet and YOLOv8s-pose

## Intro
* Augmentations
* Fine-Tuning
* Eval metrics: IoU (for bounding box), PCK@50, OKS

## Environment Setup

I am using Micromamba, here's the installation link - https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

Homebrew
```bash
brew install micromamba
```

If you want to quickly use micromamba in an ad-hoc usecase, you can run
```bash
export MAMBA_ROOT_PREFIX=/some/prefix  # optional, defaults to ~/micromamba
eval "$(./bin/micromamba shell hook -s posix)"
# Linux/bash:
./bin/micromamba shell init -s bash -p ~/micromamba  # this writes to your .bashrc file
# sourcing the bashrc file incorporates the changes into the running session.
# better yet, restart your terminal!
source ~/.bashrc

# macOS/zsh:
./micromamba shell init -s zsh -p ~/micromamba
source ~/.zshrc
```

Activate Micromamba 
```bash
micromamba activate  # this activates the base environment
```

clone the repository
```bash
https://github.com/RashaJ90/HPE-HRNet-YOLO.git
```

Create a conda environment after opening the repository
```bash
micromamba env create -f hpe.yaml
```

```bash
micromamba activate Hpe
```

Install the requirements
```bash
pip install -r requirements.txt
```

## Data
Install LSP using this Link -  https://plmlab.math.cnrs.fr/chevallier-teaching/datasets/leeds-sport-pose

* LSP Images- The Leeds Sports Pose dataset contains 9428 pose annotated images featuring people engaged in various sports activities, which introduce a wide range of poses, occlusions, and viewpoints. The images have been scaled such that the annotated person is roughly 150 pixels in length.

* README.md - this document contains the following sorted Joints:
    * Right ankle
    * Right knee
    * Right hip
    * Left hip
    * Left knee
    * Left ankle
    * Right wrist
    * Right elbow
    * Right shoulder
    * Left shoulder
    * Left elbow
    * Left wrist
    * Neck
    * Head top

* joints.mat - a MATLAB format matrix 'joints' consisting of 14 joint locations and visibility flags. Joints labelled  data points are in the same order as in the README.md file

Data Hierarchy
```bash
 data_path/
    ├── images/
    │   ├── train/   # Directory containing training images
    │   └── val/   # Directory containing validation images
    │   └── test/   # Directory containing test images[Optional]
    ├── labels/
    │   ├── train/   # Directory containing validation labels (RoboFlow annotations)
    │   └──  val/  # Directory containing training labels (RoboFlow annotations)
    └── yaml file(s)  # YAML file(s) specifying paths to images and labels:
        names:
        0: person
        path: ./path/to/data
        test: images/ train
        train: images/ val
        val: images / test
        # Keypoints
        kpt_shape: [14, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        flip_idx: [] # No need to flip index, since the joints are sorted according to model format, else fill the wanted idx  per your model format!
```
## WorkFlow

This section is made to instruct user of how to train, predict, and see results in wandb / tensoboard.

## Project Results

This section is made to add results for the best performance  

