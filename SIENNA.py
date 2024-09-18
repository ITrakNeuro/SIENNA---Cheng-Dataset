import os
import cv2
import numpy as np
import tensorflow as tf
import feature_extraction
from adversarial_training import AdversarialGenerator
from intensity_redistribution import adaptive_MRI_enhancement
from performance_evaluation import InbuiltMetricsCalculator
from sklearn.metrics import confusion_matrix

class SIENNA:

    def __init__(self, multi=False, binary=False):

            self.multi_classification()

    def multi_classification(self):
        trials,best_model,best_run,x_train, y_train, x_test, y_test,x_cv,y_cv=feature_extraction.run_hyperparameter_tunning()
        a=best_model.predict(x_cv)
        y_pred=np.argmax(a,axis=1)
        y_true=np.argmax(y_cv,axis=1)
        cm=confusion_matrix(y_true, y_pred)
        for i in range(3):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                tn = np.sum(cm) - tp - fp - fn
                accuracy = (tp + tn) / np.sum(cm)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1score = 2 * precision * recall / (precision + recall)

                print(f"\nResults for class {i}:")
                print(f"True positives: {tp}")
                print(f"False positives: {fp}")
                print(f"False negatives: {fn}")
                print(f"True negatives: {tn}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 score: {f1score}")

                

if __name__ == "__main__":
    app = SIENNA(multi=True)