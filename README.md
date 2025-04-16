# Splitting Yolo inference

## Table of Contents
1. [Overview of YOLO](#overview-of-yolo)
    - [YOLO architecture](#yolo-architecture)
2. [Challenges in Edge-Cloud Deployment](#challenges-in-edge-cloud-deployment)
    - [Splitting trade-offs and considerations](#splitting-trade-offs-and-considerations)
3. [Tools and Frameworks](#tools-and-frameworks)
    - [Docker](#docker)
    - [Data Transmission](#data-transmission)

...  
...  
...  

## Overview of YOLO

### YOLO architecture

## Challenges in Edge-Cloud Deployment

### Splitting trade-offs and considerations

## Tools and Frameworks

### Docker
The difference between the Docker images `ultralytics/ultralytics:latest` and `ultralytics/ultralytics:latest-jetson-jetpackX` lies in their target hardware, software environment, and optimizations:

| **Feature**               | **latest**                                 | **latest-jetson-jetpack6**                |
|---------------------------|--------------------------------------------|-------------------------------------------|
| **Target Hardware**       | General-purpose systems (x86_64, CUDA GPUs)| NVIDIA Jetson devices (ARM-based, aarch64)|
| **CUDA Version**          | Standard CUDA (desktop/server GPUs)        | CUDA tailored for JetPack environment     |
| **Optimization**          | General PyTorch optimizations              | TensorRT and JetPack-specific tuning      |
| **Architecture**          | x86_64                                     | ARM64 (aarch64)                           |
| **Use Case**              | Training and inference                     | Deployment on edge devices                |

### Data Transmission

