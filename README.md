# Automatic Detection of Hazardous Traffic Situations in Video using Deep Learning

This repository contains the research code, experiment outputs, and supporting material for a master’s thesis focused on the automatic detection and anticipation of dangerous traffic situations from dashcam video.

The central idea is that a collision is not only detected after it happens, but the system also tries to estimate whether the scene is becoming more dangerous before the incident starts. The project therefore formulates the task as next-frame danger estimation from video windows.

## Thesis overview

The thesis compares two complementary deep-learning approaches:

- 2D CNN + LSTM pipeline: frame-level visual features, object-level context, and temporal modeling
- 3D CNN pipeline: direct spatio-temporal learning from short video sequences

The work combines visual information with traffic-participant context, including object detection, tracking, bounding boxes, and motion-related features. The final goal is to estimate a danger score over time and identify when a driving scene becomes hazardous.

## Research scope

The project covers:

- preprocessing of dashcam video and frame extraction
- object detection and tracking of traffic participants
- optical-flow and motion modeling
- transfer learning on traffic video datasets
- frame-level and video-level evaluation of danger prediction
- assessment of detection timing and alarm behavior
- analysis of the practical limits of the proposed approach

## Repository structure

- [implementation/2D CNN](implementation/2D%20CNN/) — 2D model, dataloader, training/evaluation scripts, and experiment artifacts
- [implementation/3D CNN](implementation/3D%20CNN/) — 3D model, evaluation code, checkpoints, and generated plots
- [dataset](dataset/) — train/test split files and metadata used by the experiments
- [related_work](related_work/) — literature references used in the thesis

## Model architectures

### 2D CNN + LSTM

The 2D pipeline extracts features from individual frames and models how they evolve over time. It combines:

- a global visual stream based on a ResNet backbone
- object-level feature extraction from tracked traffic participants
- bounding-box geometry and class information
- motion cues from optical flow
- temporal modeling through LSTM and attention mechanisms

### 3D CNN

The 3D model learns spatial and temporal patterns directly from a sequence of frames. It combines:

- a 3D video backbone based on R(2+1)D-18
- object context extracted from tracked vehicles and participants
- temporal attention and feature fusion
- end-to-end learning of danger over short temporal windows

## Data and evaluation

The main dataset is DoTA (Detection of Traffic Anomaly), with additional pretraining on TAU-106K. Experiments compare the two models on frame-level and video-level tasks, including detection timing and threshold-based risk evaluation.

### Representative results

| Model | Frame-level ROC AUC (DoTA test) | Frame-level ROC AUC (extended test) |
| --- | ---: | ---: |
| 2D CNN + LSTM | 0.740 | 0.904 |
| 3D CNN | 0.775 | 0.915 |

The better-performing configuration in the thesis is the 3D model, which achieved stronger overall detection performance and better detection timing relative to the 2D baseline.

## Visual examples from the project

The repository contains evaluation and training plots that are useful for the thesis presentation and documentation.

![3D CNN ROC curve](implementation/3D%20CNN/checkpoints_3d/v24/evaluacija_DoTA/roc_curve.png)

![3D CNN training curves](implementation/3D%20CNN/checkpoints_3d/v24/training_curves.png)

## Getting started

To reproduce the experiments, the repository needs the dataset and model assets referenced in the implementations. The training and evaluation scripts are organized per model type and require the correct data paths and checkpoint locations to be configured.

Typical workflow:

1. Prepare the dataset and metadata folders.
2. Open the relevant implementation folder: [implementation/2D CNN](implementation/2D%20CNN/) or [implementation/3D CNN](implementation/3D%20CNN/).
3. Configure the YAML or script paths to match the local environment.
4. Run the training or evaluation command for the selected model.


## Thesis information

- Original title: Sustav za automatsku detekciju opasnih prometnih situacija iz videozapisa pomoću dubokog učenja
- English title: Automatic detection of hazardous traffic situations in video using deep learning
- Project type: Master’s thesis in computer engineering / software engineering

## Related literature

The [related_work](related_work/) folder contains the research papers and scientific references used to support the thesis and to contextualize the proposed traffic-danger detection approach.

## Contact

**Ivan Koturić**  
Email: [ikoturic84@gmail.com](mailto:ikoturic84@gmail.com)  
GitLab: [ikoturic](https://gitlab.com/ivan.koturic)