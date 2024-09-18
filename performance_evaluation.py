import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib
print(f"TensorFlow version: {tf.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
model = load_model("D:\\10-02-2023\\SIENNA\\optimised_model.h5")
data_dir = 'D:\10-02-2023\SIENNA\data'
x = []
y = []
z = []

class_mapping = {'N': 0, 'G': 1, 'M': 2}

for class_folder in os.listdir(data_dir):
    if class_folder in class_mapping:
        class_path = os.path.join(data_dir, class_folder)
        class_label = class_mapping[class_folder]

        for image_filename in os.listdir(class_path):
            if image_filename.lower().endswith((".png", ".jpg")):
                image_path = os.path.join(class_path, image_filename)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (240, 240))
                image = image / 255.0
                x.append(image)
                y.append(class_label)
                z.append(image_filename)

x = np.array(x)
y = np.array(y)
z = np.array(z)
y = to_categorical(y, num_classes=3)

data_dir = '/content/drive/MyDrive/adv'
x = []
y = []
z = []

class_mapping = {'N': 0, 'G': 1, 'M': 1}

for class_folder in os.listdir(data_dir):
    if class_folder in class_mapping:
        class_path = os.path.join(data_dir, class_folder)
        class_label = class_mapping[class_folder]

        for image_filename in os.listdir(class_path):
            if image_filename.lower().endswith((".png", ".jpg")):
                image_path = os.path.join(class_path, image_filename)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (240, 240))
                image = image / 255.0
                x.append(image)
                y.append(class_label)
                z.append(image_filename)

x = np.array(x)
y = np.array(y)
z = np.array(z)
y = to_categorical(y, num_classes=2)


class InbuiltMetricsCalculator:
    def __init__(self, model, x_data, y_data, image_names):
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.image_names = image_names
        self.num_classes = np.max(y_data) + 1
        self.y_pred = np.argmax(self.model.predict(self.x_data), axis=1)
        self.cm = confusion_matrix(self.y_data, self.y_pred)

    def calculate_metrics(self, class_index, threshold=0.5):
        tp = self.cm[class_index, class_index]
        fp = np.sum(self.cm[:, class_index]) - tp
        fn = np.sum(self.cm[class_index, :]) - tp
        tn = np.sum(self.cm) - tp - fp - fn
        accuracy = (tp + tn) / np.sum(self.cm)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        y_scores = self.model.predict(self.x_data)[:, class_index]
        y_pred_threshold = (y_scores >= threshold).astype(int)
        y_true_binary = (self.y_data == class_index).astype(int)
        auroc = roc_auc_score(y_true_binary, y_pred_threshold)

        result = {
            "True positives": tp,
            "False positives": fp,
            "False negatives": fn,
            "True negatives": tn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 score": f1score,
            "AUROC": auroc,
        }

        return result


    def evaluate(self, threshold=0.5, overall=False):
        results = {}

        for i in range(self.num_classes):
            class_results = self.calculate_metrics(i, threshold)
            results[f"Class {i}"] = class_results

        if overall:
            if self.num_classes == 2:
                y_scores = self.model.predict(self.x_data)[:, 1]
                y_true_binary = (self.y_data == 1).astype(int)
                auroc = roc_auc_score(y_true_binary, y_scores)
                accuracy = accuracy_score(y_true_binary, (y_scores >= threshold).astype(int))
                f1score = f1_score(y_true_binary, (y_scores >= threshold).astype(int))

                results["Overall Metrics"] = {
                    "AUROC": auroc,
                    "Accuracy": accuracy,
                    "F1 score": f1score
                }

            else:
                overall_accuracy = np.mean([metrics["Accuracy"] for metrics in results.values()])
                overall_f1_score = np.mean([metrics["F1 score"] for metrics in results.values()])

                y_true_binary = np.eye(self.num_classes)[self.y_data]
                y_scores = self.model.predict(self.x_data)
                fpr, tpr, _ = roc_curve(y_true_binary.ravel(), y_scores.ravel())
                overall_auroc = auc(fpr, tpr)

                results["Overall Metrics"] = {
                    "Accuracy": overall_accuracy,
                    "F1 score": overall_f1_score,
                    "AUROC": overall_auroc
                }
        return results



    def get_image_names(self, class_index, true_positive=False, true_negative=False, false_positive=False, false_negative=False):
        y_true_binary = (self.y_data == class_index).astype(int)
        y_pred_threshold = (self.model.predict(self.x_data)[:, class_index] >= self.threshold).astype(int)

        tp_indices = np.where((y_true_binary == 1) & (y_pred_threshold == 1))[0]
        tn_indices = np.where((y_true_binary == 0) & (y_pred_threshold == 0))[0]
        fp_indices = np.where((y_true_binary == 0) & (y_pred_threshold == 1))[0]
        fn_indices = np.where((y_true_binary == 1) & (y_pred_threshold == 0))[0]

        tp_image_names = [self.image_names[i] for i in tp_indices]
        tn_image_names = [self.image_names[i] for i in tn_indices]
        fp_image_names = [self.image_names[i] for i in fp_indices]
        fn_image_names = [self.image_names[i] for i in fn_indices]

        return {
            "True Positives": tp_image_names if true_positive else [],
            "True Negatives": tn_image_names if true_negative else [],
            "False Positives": fp_image_names if false_positive else [],
            "False Negatives": fn_image_names if false_negative else [],

        }


    def draw_plots(self, class_indices=None, auroc=False, accuracy=False, f1_score=False, recall=False, overall_auc=False):
        if class_indices is None:
            class_indices = range(self.num_classes)

        if overall_auc:
            if self.num_classes == 2:
                y_true_binary = self.y_data
                y_scores = self.model.predict(self.x_data)[:, 1]
                fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, lw=2, label=f'Overall AUROC = {roc_auc:.2f}')
            else:
                y_true_binary = np.eye(self.num_classes)[self.y_data]
                y_scores = self.model.predict(self.x_data)
                fpr, tpr, _ = roc_curve(y_true_binary.ravel(), y_scores.ravel())
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, lw=2, label=f'Overall AUROC = {roc_auc:.2f}')

        if not overall_auc or len(class_indices) > 0:
            for i in class_indices:
                y_true_binary = (self.y_data == i).astype(int)
                y_scores = self.model.predict(self.x_data)[:, i]
                fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUROC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        if accuracy:
            if self.num_classes == 2:
                class_names = ["Binary"]
                accuracies = [self.calculate_metrics(1)["Accuracy"]]
            else:
                class_names = [f'Class {i}' for i in class_indices]
                accuracies = [self.calculate_metrics(i)["Accuracy"] for i in class_indices]
            plt.figure(figsize=(10, 6))
            plt.bar(class_names, accuracies, color='skyblue')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by Class')
            plt.xticks(rotation=45)
            plt.show()

        if f1_score:
            if self.num_classes == 2:
                class_names = ["Binary"]
                f1_scores = [self.calculate_metrics(1)["F1 score"]]
            else:
                class_names = [f'Class {i}' for i in class_indices]
                f1_scores = [self.calculate_metrics(i)["F1 score"] for i in class_indices]
            plt.figure(figsize=(10, 6))
            plt.bar(class_names, f1_scores, color='lightgreen')
            plt.xlabel('Class')
            plt.ylabel('F1 Score')
            plt.title('F1 Score by Class')
            plt.xticks(rotation=45)
            plt.show()

        if recall:
            if self.num_classes == 2:
                class_names = ["Binary"]
                recalls = [self.calculate_metrics(1)["Recall"]]
            else:
                class_names = [f'Class {i}' for i in class_indices]
                recalls = [self.calculate_metrics(i)["Recall"] for i in class_indices]
            plt.figure(figsize=(10, 6))
            plt.bar(class_names, recalls, color='lightcoral')
            plt.xlabel('Class')
            plt.ylabel('Recall')
            plt.title('Recall by Class')
            plt.xticks(rotation=45)
            plt.show()

