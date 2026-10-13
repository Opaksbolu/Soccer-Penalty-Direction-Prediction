# Soccer Penalty Direction Prediction

## Project Overview

This project investigates whether deep learning can predict the
direction of a soccer penalty kick before the kicker contacts the ball.

The model will classify each penalty kick as:

- Left
- Center
- Right

All directions are defined from the goalkeeper's perspective.

## Research Question

How early before ball contact can a deep learning model reliably
predict whether a penalty kick will go left, center, or right?

## Team Members

- Abdul-Azeem Opakunle
- Zackary Butz

## Dataset

Primary dataset:
Women's Soccer Penalty Kick Data Captured by Computer Vision Models

Source:
https://data.mendeley.com/datasets/brx9bsxnpx/2

## Proposed Deep Learning Architecture

MobileNetV3 + Gated Recurrent Unit (GRU)

- MobileNetV3 extracts visual features from video frames.
- GRU learns temporal movement patterns.
- A classification layer predicts left, center, or right.

## Baseline Models

1. Majority-class classifier
2. Pose keypoints + logistic regression

## Evaluation

- Accuracy
- Precision
- Recall
- F1 score
- Macro-F1
- Confusion matrix
- Inference latency

## Prediction Horizons

The model will be evaluated using video sequences ending:

- 1.0 second before ball contact
- 0.5 seconds before ball contact
- 0.2 seconds before ball contact

## Project Status

Phase 1: Repository setup and dataset inspection.
