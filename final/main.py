import sys
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Data directory
        self.data_dir = 0
        self.image_count = 0

        # Data parameters
        self.batch_size = 32
        self.img_height = 180
        self.img_width = 180
        self.validation_split = 0.2

        """Initialize the window and display its contents to the screen."""
        self.setMinimumSize(1280, 720)
        self.setWindowTitle("Motorcycle Detector 1.0")
        self.setWindowIcon(QIcon("images/ducati.png"))

        """Set up the toolbar, input video, and output video"""
        # Setting up the content layout
        content = QHBoxLayout(self)
        self.setLayout(content)

        # Setting up the menu bar
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Creating menus using a QMenu object
        file_menu = QMenu("&File", self)
        menu_bar.addMenu(file_menu)
        tools_menu = QMenu("&Tools", self)
        menu_bar.addMenu(tools_menu)
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)

        # Creating sub menus for File
        download_sample_dataset_action = QAction("&Download sample dataset", self)
        download_sample_dataset_action.triggered.connect(self._download_sample_dataset)
        file_menu.addAction(download_sample_dataset_action)

        # Creating sub menus for Tools
        preprocessing_action = QAction("1.&Preprocess sample dataset", self)
        preprocessing_action.triggered.connect(self._preprocessing)
        tools_menu.addAction(preprocessing_action)


        # Setting up the status bar
        status_bar = QStatusBar(self)
        tensorflow_ver_lb = QLabel("TensorFlow version: " + tf.__version__)
        image_count_lb = QLabel("Image count: " + str(self.image_count))
        status_bar.addPermanentWidget(tensorflow_ver_lb)
        status_bar.addWidget(image_count_lb)
        self.setStatusBar(status_bar)

        self.show()

    def _download_sample_dataset(self):
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        self.data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        self.data_dir = pathlib.Path(self.data_dir)
        print("Download complete")
        self.image_count = len(list(self.data_dir.glob('*/*.jpg')))

    def _preprocessing(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        class_names = train_ds.class_names
        print(class_names)

        # Visualize the data
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")

        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Standardize the data
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

        # Create a model
        num_classes = 5

        model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Model summary
        model.summary()

        # Train the model
        epochs = 10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        # Visualize training results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
