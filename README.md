# SIENNA---Cheng Dataset

## Abstract

Purpose: Computer vision AI developed for MRI clinical diagnostics is typically data heavy, high parameter and architecturally complex, bringing elements less suitable for clinical applications where data is scarce and compact feature-focused architectures could excel. 
Methods: In a retrospective diagnostic comparative study we developed a compact deep learning architecture, SIENNA, and performed MRI brain tumor multiclass classification and validation against published models and datasets. SIENNA initial training applied DICOM FLAIR axial images from 17 total patients for glioblastoma and metastasized tumors under IRB protocol. A total of only 386 tumor type and non-tumor 2D scans were distributed into 60-20-20 train-validation-test (TVT) sets and multiple data-focusing strategies applied, including a region-aware intensity standardization algorithm. Performance optimizations included adversarial training to avoid shortcut learning, SMOTE class size balance and augmentation, 100-run sampling and retraining on challenging images, and hyperparameter tuning. 
Results: SIENNA was evaluated across four MRI multi-tumor datasets and multiple imaging modalities, MRI manufacturers, and field strengths. On the original dataset, SIENNA achieved mean accuracies of 90% ± 3.2% for glioblastoma, 93% ± 2.6% for metastatic, and 92% ± 5.5% for non-tumor classification across 100 randomized TVT runs, with AUROC of 95% ± 2% and F1-score of 87% ± 12%. SIENNA high performance generalized to data from BraTS 2020 (accuracy 97%, AUROC 99%), UCSF (accuracy 92%, AUROC 96%), and Cheng T1-ce (accuracy 94%, AUROC 98%) datasets.
Conclusion: SIENNA extends clinical diagnostics to small dataset applications important for disease typing, subtyping or rare disease analysis in a minimal footprint. 


This repository offers the pre-trained weights of SIENNA, specifically trained on clinical data. The python script retrieves sequentially pre-trimmed MRI scans from designated data paths, applies "Pixel Redistribution Enhancement, Masking, Optimization" (PREMO) for equalization, and predicts z-slices to be glioma, meningioma and pituitary tumors using SIENNA pre-trained architecture. The results are then presented using metrics that quantify SIENNA's performance, such as Accuracy and F1-Score.

## Dataset
To extend evaluation of SIENNA’s ability to generalize across new clinical patient data, varying tumor types, and new MR image modality, we tested SIENNA on a dataset from Cheng, Jun 2017 [Cheng, Jun (2017). brain tumor dataset. figshare. [Dataset](https://doi.org/10.6084/m9.figshare.1512427.v5)]  (referred to here as ***Cheng Dataset***) that also uses alternate data preprocessing. This brain tumor dataset contains 3064 T1-weighted contrast-enhanced images (T1-ce) from 233 patients from Nanfang Hospital and General Hospital, Tianjin Medical University, China (2005 - 2010). The subset utilized in our study consisted of 996 axial scans, distributed among meningioma (207 slices), glioma (499 slices), and pituitary tumor (290 slices). SIENNA performance achieves remarkable mean accuracies of 96.0% (Pituitary, SD=1.2%), 92.6% (Meningioma, SD=1.6%), and 95.3% (Glioma, SD=1.4%) across 100 random sub-samples. This exemplifies SIENNA’s stable precision and capacity to harness larger, heterogeneous unseen datasets, revealing generalizability for diagnostic predictions. 

## Hardware Requirements for Training

The training process was conducted on a Microsoft Windows 11 workstation equipped with an Intel(R) Core (TM) i7-10750H six-core CPU, 16 GB of system RAM, and a single NVIDIA GTX 1650 GPU boasting 4 GB of GPU RAM.

## Python Environment

Python libraries and modules required are listed in requirements.txt.

<pre>pip install -r requirements.txt </pre>


## SIENNA Architecture 

### Source code for testing SIENNA:

1. SIENNA Pre-trained.h5: Pre-trained weights of SIENNA on Cheng Dataset. 
2. SIENNA.ipynb: Reads MRI images, applies preprocessing (including the patented PREMO method), loads a pre-trained model, and evaluates its performance on the test data.
3. intensity_redistribution.py: PREMO is defined here.
4. misc.py: Miscellaneous code

## Running the SIENNA Evaluation on Clinical Dataset 2

To reproduce the results of the SIENNA model on Cheng Dataset, please follow the steps below:

1. **Upload Analysis Script to Colab**: 
- Click [here](https://github.com/ITrakNeuro/SIENNA---CD2/blob/main/SIENNA.ipynb) to download the `SIENNA.ipynb` file to your local system.
- Go to [Google Colab](https://colab.research.google.com/).
- Click on **File > Upload Notebook** and select the `SIENNA.ipynb` file you just downloaded.
  
Alternatively,

- Click [here](https://colab.research.google.com/drive/1idDhHzQh4UpvPMNW-jQ4anQbjeY4sBli?usp=sharing) to open the notebook directly in Google Colab.
- In Colab, go to **File > Save a copy in Drive** to create your own editable copy of the notebook.
   
2. **Run the Cells**: 
   - Once the Colab file opens, please run the cells sequentially to clone the repository, install dependencies, and execute the code.
   - The notebook will guide you through the steps to test the SIENNA model on Cheng Dataset and analyze its performance.

3. **View the Results**:
   - The results will include model performance metrics like accuracy, precision, recall, F1 score, and AUROC for different tumor types.
   - These metrics will be logged in an Excel sheet, which you can access and download for further analysis.

4. **Contact**: If you encounter any issues, feel free to raise them with the authors, and we will assist you.
