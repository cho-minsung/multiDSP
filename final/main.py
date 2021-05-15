import sys
import cv2
import tensorflow as tf
import time
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
from xml.dom import minidom
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from tensorflow.keras import layers
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from google_images_download import google_images_download


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.lut = {}
        self.lut["accessory"] = 0
        self.lut["top"] = 1
        self.lut["bottom"] = 2
        self.lut["bag"] = 3
        self.lut["shoes"] = 4

        # Google images download object
        self.response = google_images_download.googleimagesdownload()

        # Data directory
        self.data_dir = 0
        self.image_count = 0

        # Data parameters
        self.batch_size = 32
        self.img_height = 160
        self.img_width = 160
        self.validation_split = 0.2

        # Train and Validation dataset
        self.train_ds = 0
        self.val_ds = 0
        self.test_ds = 0

        # Class names
        self.class_names = 0

        # Normalized layer
        self.normalization_layer = 0

        # Model
        self.base_model = 0
        self.model = 0

        # History
        self.history = 0

        # Epochs
        self.epochs = 0

        """Initialize the window and display its contents to the screen."""
        self.setMinimumSize(1366, 720)
        self.setWindowTitle("Motorcycle Detector 1.0")
        self.setWindowIcon(QIcon("images/ducati.png"))

        """Set up the toolbar, input video, and output video"""
        # Setting up the content layout
        content = QHBoxLayout()
        center_widget = QWidget()
        center_widget.setLayout(content)
        self.setCentralWidget(center_widget)

        # Setting up the image layout
        image_layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setPixmap(QPixmap("images/temp.png"))
        self.label.setScaledContents(True)
        self.label.setMinimumWidth(720)
        self.label.setMaximumWidth(900)
        image_layout.addWidget(self.label)
        content.addLayout(image_layout)

        # Setting up the grid layout for buttons
        button_layout = QGridLayout()
        content.addLayout(button_layout)

        # Setting up the button layout
        preprocessing_button = QPushButton("1.&Select & Preprocess a dataset")
        preprocessing_button.clicked.connect(self._select_dataset)
        preprocessing_button.setChecked(True)
        self.preprocessing_status = QLabel()
        self.preprocessing_status.setText("Not done.")

        configuring_button = QPushButton("2.&Configuring for performance")
        configuring_button.clicked.connect(self._configure_dataset)
        configuring_button.setChecked(True)
        self.configuring_status = QLabel()
        self.configuring_status.setText("Not done.")

        model_button = QPushButton("3.&Create & compile a model")
        model_button.clicked.connect(self._create_a_model)
        model_button.setChecked(True)
        self.model_status = QLabel()
        self.model_status.setText("Not done.")

        mobilenet_button = QPushButton("3.&Load MobileNet V2")
        mobilenet_button.clicked.connect(self._use_mobile_net_v2)
        mobilenet_button.setChecked(True)
        self.mobilenet_status = QLabel()
        self.mobilenet_status.setText("Not done.")

        train_button = QPushButton("4.&Train the model")
        train_button.clicked.connect(self._train_model)
        train_button.setChecked(True)
        self.train_status = QLabel()
        self.train_status.setText("Not done.")

        load_model_button = QPushButton("4.&Load the model")
        load_model_button.clicked.connect(self._load_model)
        load_model_button.setChecked(True)
        self.load_model_status = QLabel()
        self.load_model_status.setText("Not done.")

        predict_button = QPushButton("5.&Predict an image")
        predict_button.clicked.connect(self._predict)
        predict_button.setChecked(True)
        self.predict_status = QLabel()
        self.predict_status.setText("Not done.")

        # filling up the button layout
        button_layout.addItem(QSpacerItem(30, 30), 0, 0)
        button_layout.addItem(QSpacerItem(30, 30), 0, 3)
        button_layout.setColumnStretch(5, 1)
        button_layout.addWidget(preprocessing_button, 0, 1)
        button_layout.addWidget(self.preprocessing_status, 0, 2)
        button_layout.addWidget(configuring_button, 1, 1)
        button_layout.addWidget(self.configuring_status, 1, 2)
        button_layout.addWidget(model_button, 2, 1)
        button_layout.addWidget(self.model_status, 2, 2)
        button_layout.addWidget(mobilenet_button, 2, 4)
        button_layout.addWidget(self.mobilenet_status, 2, 5)
        button_layout.addWidget(load_model_button, 3, 4)
        button_layout.addWidget(self.load_model_status, 3, 5)
        button_layout.addWidget(train_button, 3, 1)
        button_layout.addWidget(self.train_status, 3, 2)
        button_layout.addWidget(predict_button, 4, 1)
        button_layout.addWidget(self.predict_status, 4, 2)

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

        load_model_action = QAction("&Load a model", self)
        load_model_action.triggered.connect(self._load_model)
        file_menu.addAction(load_model_action)

        download_action = QAction("&Download images from Google", self)
        download_action.triggered.connect(self._download_dataset)
        file_menu.addAction(download_action)

        # Creating sub menus for Tools
        convert_action = QAction("&Convert mp4 to pngs", self)
        convert_action.triggered.connect(self._import_video)
        tools_menu.addAction(convert_action)

        xml_action = QAction("&Convert xml to csv", self)
        xml_action.triggered.connect(self._convert_xml_to_csv)
        tools_menu.addAction(xml_action)

        yolo_action = QAction("&Convert xml to yolo txt", self)
        yolo_action.triggered.connect(self.convert_xml2yolo)
        tools_menu.addAction(yolo_action)


        # Setting up the status bar
        status_bar = QStatusBar(self)
        tensorflow_ver_lb = QLabel("TensorFlow version: " + tf.__version__)
        image_count_lb = QLabel("Image count: " + str(self.image_count))
        status_bar.addPermanentWidget(tensorflow_ver_lb)
        status_bar.addWidget(image_count_lb)
        self.setStatusBar(status_bar)

        self.show()

    # function to extract bounding boxes from an annotation file
    def extract_boxes(filename):
        # load and parse the file
        tree = ET.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height


    def convert_xml2yolo(self):

        for fname in glob.glob("models/annotations/xmls/*.xml"):

            xmldoc = minidom.parse(fname)

            fname_out = (fname[:-4] + '.txt')

            with open(fname_out, "w") as f:

                itemlist = xmldoc.getElementsByTagName('object')
                size = xmldoc.getElementsByTagName('size')[0]
                width = int((size.getElementsByTagName('width')[0]).firstChild.data)
                height = int((size.getElementsByTagName('height')[0]).firstChild.data)

                for item in itemlist:
                    # get class label
                    classid = (item.getElementsByTagName('name')[0]).firstChild.data
                    if classid in self.lut:
                        label_str = str(self.lut[classid])
                    else:
                        label_str = "-1"
                        print("warning: label '%s' not in look-up table" % classid)

                    # get bbox coordinates
                    xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                    ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                    xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                    ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                    b = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bb = convert_coordinates((width, height), b)
                    # print(bb)

                    f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

            print("wrote %s" % fname_out)

    def _convert_xml_to_csv(self):
        path = QFileDialog.getExistingDirectory(caption="Select the directory of xml files.")
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv('models/annotations/labels.csv', index=None)


    def _download_dataset(self):
        keyword, ok = QInputDialog.getText(self, "Keyword input.",
                                           "Enter the keyword of the image you'd like to download.")
        if not ok:
            return
        arguments = {"keywords": keyword,
                     "format": "jpg",
                     "limit": 100,
                     "print_urls": True,
                     "size": "medium"}

        directory_name = self.response.download(arguments)
        print(directory_name)

    def _load_model(self):
        directory_name = QFileDialog.getExistingDirectory(caption="Select the directory to load a model.")
        self.model = tf.keras.models.load_model(directory_name)

        self.load_model_status.setText(str(directory_name) + "has been loaded.")

    def _import_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        video_name, _ = QFileDialog.getOpenFileName(self, caption="Select the video.", options=options,
                                                    filter="All Files (*);;MP4 files (*.mp4)")
        video = cv2.VideoCapture(video_name)
        success, image = video.read()
        count = 0
        directory_name = QFileDialog.getExistingDirectory(caption="Select the directory to save the images.")
        print(directory_name)
        name, ok = QInputDialog.getText(self, "Name convention.",
                                        "Enter the name convention of the image.")
        if not ok:
            return
        while success:
            cv2.imwrite(directory_name + "/" + name + "frame%d.png" % count, image)
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
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")

        self.val_batches = tf.data.experimental.cardinality(self.val_ds)
        self.test_ds = self.val_ds.take(self.val_batches // 5)
        self.val_ds = self.val_ds.skip(self.val_batches // 5)

        self.preprocessing_status.setText("A dataset is selected: " + str(self.class_names))

    def _configure_dataset(self):
        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=AUTOTUNE)
        self.configuring_status.setText("Auto Tune is done.")

    def _create_a_model(self):
        # Create a model
        num_classes, ok = QInputDialog.getInt(self, "Classes input.",
                                              "Enter the number of classes.",
                                              5)
        if not ok:
            return

        data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(self.img_height,
                                                                          self.img_width,
                                                                          3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

        self.model = tf.keras.Sequential([
            data_augmentation,
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
        self.model_status.setText("Model is created.")
        self.model.summary()

    def _train_model(self):
        # Train the model
        self.epochs, ok = QInputDialog.getInt(self, "Epoch number input.",
                                              "Enter the number of epoch.",
                                              5)
        if not ok:
            return

        start_time = time.time()

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        self.train_status.setText("%s seconds to train the model." % (time.time() - start_time))

        # Save the model
        self.model.save('model')

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

    def _use_mobile_net_v2(self):

        data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(self.img_height,
                                                                          self.img_width,
                                                                          3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        print("Preprocessing is done.")
        rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

        IMG_SHAPE = (self.img_height, self.img_width) + (3,)

        self.base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        print("Building a base model.")

        image_batch, label_batch = next(iter(self.train_ds))
        feature_batch = self.base_model(image_batch)
        print("Feature batch shape:", feature_batch.shape)

        self.base_model.trainable = True
        for layer in self.base_model.layers[:100]:
            layer.trainable = False

        self.base_model.summary()

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print("feature batch average:", feature_batch_average.shape)

        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)
        print("prediction batch shape:", prediction_batch.shape)

        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)

        base_learning_rate = 0.0001
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
                      metrics=['accuracy'])

        self.mobilenet_status.setText("MobileNet is loaded.")
        self.model.summary()

    def _predict(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_name, _ = QFileDialog.getOpenFileName(self, caption="Select the image to predict.", options=options,
                                                    filter="All Files (*);;png files (*.png)")
        self.label.setPixmap(QPixmap(image_name))

        img = tf.keras.preprocessing.image.load_img(
            image_name, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        self.predict_status.setText("This image most likely belongs to {} with a {:.2f} percent confidence."
                                    .format(self.class_names[np.argmax(score)], 100 * np.max(score)))

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

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
