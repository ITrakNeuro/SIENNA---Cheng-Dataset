# SIENNA---CD2

## Abstract
Contemporary machine learning models for computer vision, although abundant, are largely inappropriate for clinical diagnostics. Clinical sophistication must address data consistency, avoid large parametric needs to reduce model complexity, and achieve stable generalizability across new patient data. Here, we achieve these goals in SIENNA a “Lightweight Energy-efficient Adaptive Next generation” artificial intelligence (LEAN AI) platform along with development of new algorithms for DICOM data consistency and approaches for improved integration of clinical data with deep learning architectures. Applied in the context of brain tumor diagnostics, SIENNA is a nimble AI that  requires 175K-285K trainable parameters, 122X less in comparison to other state-of-the-art AI ML tumor models, while outperforming these models. SIENNA is generalizable across diverse patient datasets in inductive tests on benchmark and clinical datasets, achieving high average accuracies of 93-96% in three-way multiclass classification of MRI tumor data, across mixed 1.5 and 3.0 Tesla data and machines. We apply no DICOM MRI data preprocessing beyond data consistency while achieving a parameter-efficient generalizable ML pipeline. SIENNA demonstrates that small clinical datasets can be sufficient to design robust clinical ready architectures to facilitate expanded ML applications in multimodal data integration in a wider range of clinical diagnostic tasks.

![alt text](https://github.com/ITrakNeuro/SIENNA-Nature/blob/main/Comparitive%20Study.png)

This repository offers the pre-trained weights of SIENNA, specifically trained on clinical data. The python script retrieves sequentially pre-trimmed MRI scans from designated data paths, applies "Pixel Redistribution Enhancement, Masking, Optimization" (PREMO) for equalization, and predicts z-slices to be glioma, meningioma and pituitary tumors using SIENNA pre-trained architecture. The results are then presented using metrics that quantify SIENNA's performance, such as Accuracy and F1-Score.

## Dataset
To extend evaluation of SIENNA’s ability to generalize across new clinical patient data, varying tumor types, and new MR image modality, we tested SIENNA on a dataset from Cheng, Jun 2017 [Cheng, Jun (2017). brain tumor dataset. figshare. [Dataset](https://doi.org/10.6084/m9.figshare.1512427.v5)]  (referred to here as ***Clinical Dataset 2***) that also uses alternate data preprocessing. This brain tumor dataset contains 3064 T1-weighted contrast-enhanced images (T1-ce) from 233 patients from Nanfang Hospital and General Hospital, Tianjin Medical University, China (2005 - 2010). The subset utilized in our study consisted of 996 axial scans, distributed among meningioma (207 slices), glioma (499 slices), and pituitary tumor (290 slices). SIENNA performance achieves remarkable mean accuracies of 96.0% (Pituitary, SD=1.2%), 92.6% (Meningioma, SD=1.6%), and 95.3% (Glioma, SD=1.4%) across 100 random sub-samples. This exemplifies SIENNA’s stable precision and capacity to harness larger, heterogeneous unseen datasets, revealing generalizability for diagnostic predictions. 

## Hardware Requirements for Training

The training process was conducted on a Microsoft Windows 11 workstation equipped with an Intel(R) Core (TM) i7-10750H six-core CPU, 16 GB of system RAM, and a single NVIDIA GTX 1650 GPU boasting 4 GB of GPU RAM.

## Python Environment

Python libraries and modules required are listed in requirements.txt.

<pre>pip install -r requirements.txt </pre>


## SIENNA Architecture 

### Source code for testing SIENNA:

1. SIENNA Pre-trained.h5: Pre-trained weights of SIENNA on Clinical Data 2. 
2. SIENNA.ipynb: Reads MRI images, applies preprocessing (including the patented PREMO method), loads a pre-trained model, and evaluates its performance on the test data.
3. intensity_redistribution.py: PREMO is defined here.
4. misc.py: Miscellaneous code

## Running the SIENNA Evaluation on Clinical Dataset 2

To reproduce the results of the SIENNA model on Clinical Dataset 2, please follow the steps below:

1. **Upload Analysis Script to Colab**: 
- Click [here](https://github.com/ITrakNeuro/SIENNA---CD2/blob/main/SIENNA.ipynb) to download the `SIENNA.ipynb` file to your local system.
- Go to [Google Colab](https://colab.research.google.com/).
- Click on **File > Upload Notebook** and select the `SIENNA.ipynb` file you just downloaded.
  
Alternatively,

- Click [here](https://colab.research.google.com/drive/1idDhHzQh4UpvPMNW-jQ4anQbjeY4sBli?usp=sharing) to open the notebook directly in Google Colab.
- In Colab, go to **File > Save a copy in Drive** to create your own editable copy of the notebook.
   
2. **Run the Cells**: 
   - Once the Colab file opens, please run the cells sequentially to clone the repository, install dependencies, and execute the code.
   - The notebook will guide you through the steps to test the SIENNA model on Clinical Dataset 2 and analyze its performance.

3. **View the Results**:
   - The results will include model performance metrics like accuracy, precision, recall, F1 score, and AUROC for different tumor types.
   - These metrics will be logged in an Excel sheet, which you can access and download for further analysis.

4. **Contact**: If you encounter any issues, feel free to raise them with the authors, and we will assist you.
