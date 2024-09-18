import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# Key Class: AdversarialGenerator
# This class facilitates the generation and training of adversarial examples for a given SIENNA model.
# It can be used to create adversarial examples, visualize perturbation maps, and evaluate the model's robustness.
# Main Features:
# 1. `__init__`: Initializes the AdversarialGenerator with the SIENNA model, original images, true labels, loss function,
#    perturbation magnitude (eps), and other optional parameters for visualization.
# 2. `generate`: Generates adversarial examples by applying perturbations to the original images based on the loss gradient.
# 3. `_plot_perturbation`: Visualizes perturbation maps for adversarial examples.
# 4. `_plot_example`: Plots original and adversarial examples for visual inspection.
# Additional Features:
# - `adversarial_training`: Performs adversarial training by optimizing the model using generated adversarial examples
#    for a specified number of iterations.
# - `evaluate_adversarial_examples`: Evaluates the model's accuracy on the generated adversarial examples.

class AdversarialGenerator:
    def __init__(self, model, x, y, loss_function, eps=0.3, stabilization_factor=1e-8, plot_perturbation=False, plot_example=False):
        """
        Initialize the AdversarialGenerator.

        Args:
            model: SIENNA model.
            x: original images for which adversarial examples will be generated.
            y: True labels corresponding to the original images.
            loss_function: Loss function to optimize during adversarial example generation.
            eps: Perturbation magnitude.
            stabilization_factor: Perturbation Normalization constant.
            plot_perturbation: Whether to plot the generated perturbation maps.
            plot_example: Whether to plot original and adversarial examples.
        """

        self.model = model
        self.x = x
        self.y = y
        self.loss_function = loss_function
        self.eps = eps
        self.stabilization_factor = stabilization_factor
        self.plot_perturbation = plot_perturbation
        self.plot_example = plot_example

    def generate(self):

        x_adv = tf.identity(self.x) # Create a copy of images.
        y_adv = tf.identity(self.y) # Create a copy of labels.

        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            y_pred = self.model(x_adv, training=False) # Make predictions on the present model.
            loss = self.loss_function(y_adv, y_pred)   # Calculate loss.


        gradients = tape.gradient(loss, x_adv)    # Gradient Computation.

        gradients /= (tf.math.reduce_std(gradients) + self.stabilization_factor)    # Gradient Normalization.

        x_adv += self.eps * gradients      # Apply perturbations to x_adv.

        if self.plot_perturbation:
            self._plot_perturbation(gradients)
        if self.plot_example:
            self._plot_example(self.x, x_adv)

        return x_adv, y_adv

    def _plot_perturbation(self, gradients):
        # Plot the perturbation maps for adversarial examples

        num_images = gradients.shape[0]
        plt.figure(figsize=(num_images * 5, 5))
    
        for i in range(num_images):
            noise = np.clip(gradients[i], -1, 1)
            plt.subplot(1, num_images, i + 1)
            plt.imshow(noise, cmap='gray' if noise.ndim == 2 else None)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def _plot_example(self, original, adversarial):
        # Plot original and adversarial examples.

        num_images = original.shape[0]
        plt.figure(figsize=(num_images * 10, 10))

        for i in range(num_images):
            original_image = np.clip(original[i], 0, 1)
            plt.subplot(2, num_images, i + 1)
            plt.imshow(original_image)
            plt.axis('off')

            adversarial_image = np.clip(adversarial[i], 0, 1)
            plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(adversarial_image)
            plt.axis('off')

        plt.tight_layout()
        plt.show()


class AdversarialGenerator:
    def __init__(self, model, x, y, loss_function, eps=0.3, stabilization_factor=1e-8, plot_perturbation=False, plot_example=False, training_iterations=1):

        """
        Initialize the AdversarialGenerator.

        Args:
            model: SIENNA model.
            x: original images for which adversarial examples will be generated.
            y: True labels corresponding to the original images.
            loss_function: Loss function to optimized.
            eps: Perturbation magnitude.
            stabilization_factor: Small constant for gradient normalization.
            plot_perturbation: Whether to plot the generated perturbation maps.
            plot_example: Whether to plot original and adversarial examples.
            training_iterations: Number of adversarial training iterations.
        """

        self.model = model
        self.x = x
        self.y = y
        self.loss_function = loss_function
        self.eps = eps
        self.stabilization_factor = stabilization_factor
        self.plot_perturbation = plot_perturbation
        self.plot_example = plot_example
        self.training_iterations = training_iterations

    def adversarial_training(self, optimizer):
        for iteration in range(self.training_iterations):
            print(f"Training iteration: {iteration+1}")

            x_adv, y_adv = self.generate() # Generate examples of current iter.

            with tf.GradientTape() as tape:
                predictions = self.model(x_adv, training=True)  # Predict prediction on current model.
                loss = self.loss_function(y_adv, predictions) # Compute Loss.

            grads = tape.gradient(loss, self.model.trainable_variables)  # Gradient Computation
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.plot_perturbation:
            self._plot_perturbation(self.gradients)
        if self.plot_example:
            self._plot_example(self.x, x_adv)

    def generate(self):

        x_adv = tf.identity(self.x)  # Create copy of x.
        y_adv = tf.identity(self.y)  # Create copy of y.

        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            y_pred = self.model(x_adv, training=False)  # Model prediction.
            loss = self.loss_function(y_adv, y_pred)    # Loss Computation.

        self.gradients = tape.gradient(loss, x_adv)  # Gradient Computation
        self.gradients /= (tf.math.reduce_std(self.gradients) + self.stabilization_factor)  # Gradient Normalization.
        x_adv += self.eps * self.gradients           # Apply perturbations to x_adv.

        return x_adv, y_adv

    def evaluate_adversarial_examples(self, x_adv, y_adv):
        predictions = self.model.predict(x_adv)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_adv, axis=1))
        print(f'Accuracy on adversarial examples: {accuracy:.2%}')

    def _plot_perturbation(self, gradients):
      # Plot the perturbation maps for adversarial examples

        num_images = gradients.shape[0]
        plt.figure(figsize=(num_images * 5, 5))

        for i in range(num_images):
            noise = np.clip(gradients[i], -1, 1)
            plt.subplot(1, num_images, i + 1)
            plt.title(f'Noise {i+1}')
            plt.imshow(noise, cmap='gray' if noise.ndim == 2 else None)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def _plot_example(self, original, adversarial):
      # Plot original and adversarial examples.

        num_images = original.shape[0]
        plt.figure(figsize=(num_images * 10, 10))

        for i in range(num_images):
            original_image = np.clip(original[i], 0, 1)
            plt.subplot(2, num_images, i + 1)
            plt.title(f'Original {i+1}')
            plt.imshow(original_image)
            plt.axis('off')

            adversarial_image = np.clip(adversarial[i], 0, 1)
            plt.subplot(2, num_images, num_images + i + 1)
            plt.title(f'Adversarial {i+1}')
            plt.imshow(adversarial_image)
            plt.axis('off')

        plt.tight_layout()
        plt.show()