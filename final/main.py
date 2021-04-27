import sys
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import *
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

        # Train and Validation dataset
        self.train_ds = 0
        self.val_ds = 0

        # Class names
        self.class_names = 0

        # Normalized layer
        self.normalization_layer = 0

        # Model
        self.model = 0

        # History
        self.history = 0

        # Epochs
        self.epochs = 0

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

        convert_action = QAction("&Convert mp4 to pngs", self)
        convert_action.triggered.connect(self._import_video)
        file_menu.addAction(convert_action)

        load_model_action = QAction("&Load a model", self)
        load_model_action.triggered.connect(self._load_model)
        file_menu.addAction(load_model_action)

        # Creating sub menus for Tools
        preprocessing_action = QAction("1.&Select & Preprocess a dataset", self)
        preprocessing_action.triggered.connect(self._select_dataset)
        tools_menu.addAction(preprocessing_action)

        configuring_action = QAction("2.&Configuring for performance", self)
        configuring_action.triggered.connect(self._configure_dataset)
        tools_menu.addAction(configuring_action)

        standardize_action = QAction("3.&Standardizing the data", self)
        standardize_action.triggered.connect(self._standardize_data)
        tools_menu.addAction(standardize_action)

        model_action = QAction("4.&Create & compile a model", self)
        model_action.triggered.connect(self._create_a_model)
        tools_menu.addAction(model_action)

        train_action = QAction("5.&Train the model", self)
        train_action.triggered.connect(self._train_model)
        tools_menu.addAction(train_action)

        predict_action = QAction("6.&Predict an image", self)
        predict_action.triggered.connect(self._predict)
        tools_menu.addAction(predict_action)

        # Setting up the status bar
        status_bar = QStatusBar(self)
        tensorflow_ver_lb = QLabel("TensorFlow version: " + tf.__version__)
        image_count_lb = QLabel("Image count: " + str(self.image_count))
        status_bar.addPermanentWidget(tensorflow_ver_lb)
        status_bar.addWidget(image_count_lb)
        self.setStatusBar(status_bar)

        self.show()

    def _load_model(self):
        directory_name = QFileDialog.getExistingDirectory(caption="Select the directory to load a model.")
        self.model = tf.keras.models.load_model(directory_name)

    def _import_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        video_name, _ = QFileDialog.getOpenFileName(self, caption="Select the video.", options=options, filter=
                                               "All Files (*);;MP4 files (*.mp4)")
        video = cv2.VideoCapture(video_name)
        success, image = video.read()
        count = 0
        directory_name = QFileDialog.getExistingDirectory(caption="Select the directory to save the images.")
        print(directory_name)
        while success:
            cv2.imwrite(directory_name + "/0frame%d.png" % count, image)
            success, image = video.read()
            count += 1
        print("Conversion successful")

    def _download_sample_dataset(self):
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        self.data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        self.data_dir = pathlib.Path(self.data_dir)
        print("Download complete")
        self.image_count = len(list(self.data_dir.glob('*/*.jpg')))

    def _select_dataset(self):
        self.data_dir = QFileDialog.getExistingDirectory(caption="Select the directory for preprocessing")
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        print("A dataset is selected.")
        self.class_names = self.train_ds.class_names
        print(self.class_names)

    def _configure_dataset(self):
        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        print("Auto Tune is done.")

    def _standardize_data(self):
        # Standardize the data
        self.normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
        print("Standardizing is done.")

    def _create_a_model(self):
        # Create a model
        num_classes = 5

        self.model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
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
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        # Model summary
        self.model.summary()

    def _train_model(self):
        # Train the model
        self.epochs = 4
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        # Save the model
        self.model.save('final/')

        # Visualize training results
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

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

    def _predict(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_name, _ = QFileDialog.getOpenFileName(self, caption="Select the image to predict.", options=options,
                                                    filter="All Files (*);;png files (*.png)")
        img = tf.keras.preprocessing.image.load_img(
            image_name, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )

    def _to_be_separated(self):
        # Visualize the data
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")

        for image_batch, labels_batch in self.train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        # Predict the image
        sunflower_url = "https://media.discordapp.net/attachments/568165469115383810/832131144022622208/DzVE5ITVsAELcu7.jpg"
        sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
