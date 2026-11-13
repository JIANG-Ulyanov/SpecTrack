# SpecTrack

Official implementation of **Recover and Refine: Unifying Multimodal Tracking via Generative Versatile-Space Alignment and Spectral Expert Cascades**.

## Introduction

SpecTrack is a unified generative framework for multimodal visual tracking, supporting RGB-T, RGB-D, and RGB-E tracking.

It consists of two key components.

- **GVSA (Generative Versatile-Space Alignment)** for iterative cross-modal feature reconstruction and alignment under modality absence.
- **SpecCoE (Spectral Chain-of-Experts)** for adaptive frequency-domain feature refinement through cascaded experts.

## Datasets

SpecTrack supports the following multimodal tracking benchmarks.

- RGB-T: LasHeR, RGBT234
- RGB-D: DepthTrack, VOT2022-RGBD
- RGB-E: VisEvent

Please refer to the corresponding dataset websites for data access and preparation.

## Code Availability

This repository currently provides the key source code of SpecTrack, including the core implementations of GVSA and SpecCoE. The complete project will be released in a future update.

