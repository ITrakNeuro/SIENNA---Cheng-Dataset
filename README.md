# SIENNA---CD2

## Abstract
Contemporary machine learning models for computer vision, although abundant, are largely inappropriate for clinical diagnostics. Clinical sophistication must address data consistency, avoid large parametric needs to reduce model complexity, and achieve stable generalizability across new patient data. Here, we achieve these goals in SIENNA a “Lightweight Energy-efficient Adaptive Next generation” artificial intelligence (LEAN AI) platform along with development of new algorithms for DICOM data consistency and approaches for improved integration of clinical data with deep learning architectures. Applied in the context of brain tumor diagnostics, SIENNA is a nimble AI that  requires 175K-285K trainable parameters, 122X less in comparison to other state-of-the-art AI ML tumor models, while outperforming these models. SIENNA is generalizable across diverse patient datasets in inductive tests on benchmark and clinical datasets, achieving high average accuracies of 93-96% in three-way multiclass classification of MRI tumor data, across mixed 1.5 and 3.0 Tesla data and machines. We apply no DICOM MRI data preprocessing beyond data consistency while achieving a parameter-efficient generalizable ML pipeline. SIENNA demonstrates that small clinical datasets can be sufficient to design robust clinical ready architectures to facilitate expanded ML applications in multimodal data integration in a wider range of clinical diagnostic tasks.

![alt text](https://github.com/ITrakNeuro/SIENNA-Nature/blob/main/Comparitive%20Study.png)

This repository offers the pre-trained weights of SIENNA, specifically trained on clinical data. The python script retrieves sequentially pre-trimmed MRI scans from designated data paths, applies "Pixel Redistribution Enhancement, Masking, Optimization" (PREMO) for equalization, and predicts z-slices to be GBM, MET or tumor-free using SIENNA pre-trained architecture. The patient-specific results are then presented using metrics that quantify SIENNA's performance, such as Accuracy and F1-Score.
## Hardware Requirements for Training

The training process was conducted on a Microsoft Windows 11 workstation equipped with an Intel(R) Core (TM) i7-10750H six-core CPU, 16 GB of system RAM, and a single NVIDIA GTX 1650 GPU boasting 4 GB of GPU RAM.

## Python Environment

Python libraries and modules required are listed in requirements.txt.

<pre>pip install -r requirements.txt </pre>


## SIENNA Architecture 

### Source code for testing SIENNA:

1. SIENNA Pre-trained.h5: Pre-trained weights of SIENNA on Clinical Data 2. 
2. SIENNA analysis.py: Reads MRI images, applies preprocessing (including the patented PREMO method), loads a pre-trained model, and evaluates its performance on the test data.
3. intensity_redistribution.py: PREMO is defined here.
4. misc.py: Miscellaneous code 
