import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Key Class: adaptive_MRI_enhancement
# This class provides methods for MRI intensity equalization and enhancement, focusing on PREMO, SSIM, and histogram analysis.

# Main Features:
# 1. `__init__`: Initializes the intensity equalizer with the input image and optional parameters for analysis.
# 2. `PREMO`: Implements Pixel Redistribution Enhancement, Masking, Optimization (PREMO) for MRI intensity equalization.
# 3. `ssim`: Computes the Structural Similarity Index (SSIM) between two images to quantify their similarity.
# 4. `plot_histograms`: Plots histograms of original and equalized scans for visual comparison.
# 5. `mutual_information`: Computes Mutual Information between two images.
# 6. `entropy`: Computes the entropy of an image.
# 7. `ambe`: Computes the Absolute Mutual Information-based Enhancement (AMBE) between two images.
# 8. `rgb_to_lab`: Converts an RGB image to the LAB color space.
# 9. `traditional_HE`: Applies traditional Histogram Equalization to the LAB channel.
# 10. `adaptive_HE`: Applies Adaptive Histogram Equalization to the LAB channel.
# 11. `contrast_HE`: Applies Contrast Stretching to the LAB channel.
# 12. `clahe_HE`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.

class adaptive_MRI_enhancement:

  def __init__(self,image,comparitive_study=False,plot_hist=False):

        """
        Initialize the intensity equalizer.

        Args:
            image: scan to be equalized.
            comparitive_study: Whether to plot comparitive study.
            plot_hist: plot histograms of equalized scans.
        """
        self.image=image
        self.comparitive_study=comparitive_study
        self.plot_hist=plot_hist

  def PREMO(self,clip_limit=1.5,gamma=3):
        """
        Pixel Redistribution Enhancement, Masking, Optimization(PREMO) for MRI intensity equalization
        Args:
            image: numpy array
            clip_limit: pixel threshold to clip
            gamma: gamma correction parameter
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        #otsu's thresholding
        thresh_val, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        #noise removal using morhological operations
        mask = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=3)
        #cumulative distribution function (CDF)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
        #mapping intensities to CDF
        mapping = np.interp(np.arange(256), np.arange(256), cdf)
        out = np.interp(gray.flatten(), np.arange(256), mapping).reshape(gray.shape)
        out = np.clip(out, 0, 255)
        if clip_limit > 0:
            hist, _ = np.histogram(out.flatten(), bins=256, range=(0, 255))
            excess = np.sum(hist) - clip_limit * gray.size
            if excess > 0:
                limit_val = np.argmax(hist.cumsum() > excess)
                out = np.clip(out, 0, limit_val)
        # Gamma Correction
        out = (out - np.min(out)) / (np.max(out) - np.min(out)) * 255
        out = np.power(out / 255.0, gamma) * 255.0
        #binary mask
        out = np.where(mask == 0, gray, out)
        #intensity range 0-255
        out = np.clip(out, 0, 255)
        out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if self.plot_hist:
            self.plot_histograms(out)
        return out
  
  def ssim(self,img1, img2):
        """
        SSIM index to quanitfy similarity between two images
        """
        img1_gray,img2_gray = np.mean(img1, axis=2), np.mean(img2, axis=2)
        mean1,mean2 = np.mean(img1_gray), np.mean(img2_gray)
        sigma1,sigma2 = np.std(img1_gray), np.std(img2_gray)
        covariance = np.cov(img1_gray.ravel(), img2_gray.ravel())
        ssim = (2 * mean1 * mean2 + 0.01) * (2 * covariance[0, 1] + 0.03) / ((mean1 ** 2 + mean2 ** 2 + 0.01) * (sigma1 ** 2 + sigma2 ** 2 + 0.03))
        return ssim

  def plot_histograms(self,out):
        """
        Function to plot histograms of scans before and after equalisation
        """       
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 8))
        ssim = self.ssim(self.image, out)
        ax[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('Original')
        ax[0, 0].grid(False)
        ax[0, 1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax[0, 1].set_title(f'Equalised Scan (SSIM={ssim })')
        ax[0, 1].grid(False)
        hist_original, bins_original, _ = ax[1, 0].hist(self.image.ravel(), bins=256, range=[6, 255], color='#BB0707')
        ax[1, 0].set_ylabel('Frequency')
        ax[1, 0].set_xlabel('Pixel Values')
        ax[1, 0].text(6, -max(hist_original) * 0.12, 'Dark Pixels', color='black', ha='center')
        ax[1, 0].text(255, -max(hist_original) * 0.12, 'Bright Pixels', color='black', ha='center')
        hist_clahe, bins_clahe, _ = ax[1, 1].hist(out.ravel(), bins=256, range=[6, 255], color='#244FA2')
        ax[1, 1].set_ylabel('Frequency')
        ax[1, 1].set_xlabel('Pixel Values')
        ax[1, 1].text(6, -max(hist_clahe) * 0.12, 'Dark Pixels', color='black', ha='center')
        ax[1, 1].text(255, -max(hist_clahe) * 0.12, 'Bright Pixels', color='black', ha='center')
        plt.rcParams['axes.facecolor'] = '#D0E2EF'  
        plt.rcParams['axes.grid'] = True  
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['grid.color'] = 'white' 
        plt.tight_layout()
        plt.show()

  def mutual_information(self,img1, img2):
        joint_histogram = np.zeros((256, 256))
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                joint_histogram[img1[i,j], img2[i,j]] += 1
        joint_histogram /= img1.size
        joint_prob = joint_histogram / np.sum(joint_histogram)
        prob1, prob2= np.sum(joint_prob, axis=1), np.sum(joint_prob, axis=0)
        mutual_info = 0
        for i in range(256):
            for j in range(256):
                if joint_prob[i,j] > 0:
                    mutual_info += joint_prob[i,j] * np.log(joint_prob[i,j] / (prob1[i] * prob2[j]))
        return mutual_info

  def entropy(self,img):
        histogram = np.zeros(256)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                histogram[img[i,j]] += 1
        histogram /= img.size
        prob = histogram / np.sum(histogram)
        entropy = 0
        for i in range(256):
            if prob[i] > 0:
                entropy -= prob[i] * np.log(prob[i])

        return entropy

  def ambe(self,img1, img2):
        mi = self.mutual_information(img1, img2)
        ent1, ent2= self.entropy(img1),self.entropy(img2)
        ambe = 2 * mi / (ent1 + ent2)
        return ambe
  
  def rgb_to_lab(self,img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = L / 255
        return L, A, B
  
  def traditional_HE(self):
        l, a, b = self.rgb_to_lab(self.image)
        hist,_= np.histogram(l.flatten(),bins=256, range=(0, 255))
        cdf = hist.cumsum()
        cdf_m = (cdf - cdf.min())*255/(cdf.max()-cdf.min())
        img2 = cdf_m[l]
        hist,_ = np.histogram(img2.flatten(),bins=255,range=(0,255))
        cdf_2 = hist.cumsum()
        updated_img = cv2.cvtColor(cv2.merge((img2, a, b)), cv2.COLOR_LAB2BGR)
        return updated_img
        
  def adaptive_HE(self):
        l, a, b = self.rgb_to_lab(self.image)
        img_adapteq = exposure.equalize_adapthist(l, clip_limit=0.1)
        img_adapteq = (img_adapteq *255).astype('uint8')
        updated_lab_img2 = cv2.merge((img_adapteq,a,b))
        img_adapteq = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
        return img_adapteq
  
  def contrast_HE(self):
        l, a, b = self.rgb_to_lab(self.image)
        p2, p92 = np.percentile(l, (2, 92))
        img_rescale = exposure.rescale_intensity(l, in_range=(p2, p92))
        updated_lab_img2 = cv2.merge((img_rescale,a,b))
        contrast_stretch = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
        return contrast_stretch

  def clahe_HE(self):
       clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
       clahe_img1 = clahe.apply(self.image)
       return clahe_img1
