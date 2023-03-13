import cv2
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


class reconize_faces():
    def __init__(self):
        self.headshots_folder_name = 'Headshots'
        # dimension of images
        self.image_width, self.image_height = 224, 224
        # for detecting faces
        self.facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # set the directory containing the images
        self.images_dir = os.path.join(".", self.headshots_folder_name)
        self.current_id = 0
        self.label_ids = {}

    def detected_faces(self):
        # iterates through all the files in each subdirectories
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                    # path of the image
                    path = os.path.join(root, file)

                    # get the label name (name of the person)
                    label = os.path.basename(root).replace(" ", ".").lower()

                    # add the label (key) and its number (value)
                if not label in self.label_ids:
                    self.label_ids[label] = self.current_id
                    self.current_id += 1

                # load the image
                imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
                image_array = np.array(imgtest, "uint8")

                # get the faces detected in the image
                faces = self.facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

                # if not exactly 1 face is detected, skip this photo
                if len(faces) != 1:
                    print(f'---Photo skipped---\n')
                # remove the original image
                os.remove(path)
                # continue

                # save the detected face(s) and associate
                # them with the label
                for (x_, y_, w, h) in faces:
                    # draw the face detected
                    face_detect = cv2.rectangle(imgtest,
                            (x_, y_),
                            (x_+w, y_+h),
                            (255, 0, 255), 2)
                    plt.imshow(face_detect)
                    plt.show()

                    # resize the detected face to 224x224
                    size = (self.image_width, self.image_height)

                    # detected face region
                    roi = image_array[y_: y_ + h, x_: x_ + w]

                    # resize the detected head to target size
                    resized_image = cv2.resize(roi, size)
                    image_array = np.array(resized_image, "uint8")

                    # remove the original image
                    # os.remove(path)

                    # replace the image with only the face
                    im = Image.fromarray(image_array)
                    im.save(path)

    def train_datagen(self):
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        self.train_generator = train_datagen.flow_from_directory(
        './Headshots',
        target_size=(224,224),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

        self.train_generator.class_indices.values()
        # dict_values([0, 1, 2])
        self.NO_CLASSES = len(self.train_generator.class_indices.values())

    def model_26layers(self):
        self.base_model_26layers = VGGFace(include_top=True,
                             model='vgg16',
                             input_shape=(224, 224, 3))
        self.base_model_26layers.summary()
        # print(len(self.base_model_26layers))
        # 26 layers in the original VGG-Face

    def model_19layers(self):
        self.base_model_19layers = VGGFace(include_top=False,
                             model='vgg16',
                             input_shape=(224, 224, 3))
        # self.base_model_19layers.summary()
        # print(len(self.base_model_19layers))
        # 19 layers after excluding the last few layers

    def FC_model(self):
        x = self.base_model_19layers.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        # final layer with softmax activation
        self.preds = Dense(self.NO_CLASSES, activation='softmax')(x)
        self.model = Model(inputs=self.base_model_19layers.input, outputs=self.preds)
        self.model.summary()
        # don't train the first 19 layers - 0..18
        for layer in self.model.layers[:19]:
            layer.trainable = False
        # train the rest of the layers - 19 onwards
        for layer in self.model.layers[19:]:
            layer.trainable = True

        self.model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.fit(self.train_generator,
                  batch_size=1,
                  verbose=1,
                  epochs=20)
        # creates a HDF5 file

    def save_model(self):
        self.model.save(
            'transfer_learning_trained' +
            '_face_cnn_model.h5')

    def save_training_labels(self):
        self.class_dictionary = self.train_generator.class_indices
        self.class_dictionary = {value: key for key, value in self.class_dictionary.items()}
        print(self.class_dictionary)
        # save the class dictionary to pickle
        face_label_filename = 'face-labels.pickle'
        with open(face_label_filename, 'wb') as f:
            pickle.dump(self.class_dictionary, f)

    def open_model(self):
        # deletes the existing model
        # del self.model

        # returns a compiled model identical to the previous one
        self.model = load_model(
            'transfer_learning_trained' +
            '_face_cnn_model.h5')

    def testing_the_trained_model(self):
        # dimension of images
        self.image_width = 224
        self.image_height = 224

        # load the training labels
        face_label_filename = 'face-labels.pickle'
        with open(face_label_filename, "rb") as f:
            class_dictionary = pickle.load(f)

        self.class_list = [value for _, value in class_dictionary.items()]
        print(self.class_list)

    def predict(self):
        for i in range(1, 6):
            test_image_filename = f'./facetest/face{i}.jpg'
            print(test_image_filename)
        # load the image
            imgtest = cv2.imread(test_image_filename, cv2.IMREAD_COLOR)
            image_array = np.array(imgtest, "uint8")

        # get the faces detected in the image
            faces = self.facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

        # if not exactly 1 face is detected, skip this photo
            if len(faces) != 1:
                print(f'---We need exactly 1 face; photo skipped - --')
                print()
                continue

            for (x_, y_, w, h) in faces:
                # draw the face detected
                face_detect = cv2.rectangle(imgtest, (x_, y_), (x_ + w, y_ + h), (255, 0, 255), 2)
                plt.imshow(face_detect)
                plt.show()
                plt.savefig(f'{i}.png')

                # resize the detected face to 224x224
                size = (self.image_width, self.image_height)
                roi = image_array[y_: y_ + h, x_: x_ + w]
                resized_image = cv2.resize(roi, size)

                # prepare the image for prediction
                x = image.img_to_array(resized_image)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=1)

                # making prediction
                predicted_prob = self.model.predict(x)
                print(predicted_prob)
                print(predicted_prob[0].argmax())
                print("Predicted face: " + self.class_list[predicted_prob[0].argmax()])
                print("============================\n")

if __name__ == '__main__':
    reconize_faces = reconize_faces()
    # # # # reconize_faces.detected_faces()
    # reconize_faces.train_datagen()
    # # # # reconize_faces.model_26layers()
    # reconize_faces.model_19layers()
    # reconize_faces.FC_model()
    # reconize_faces.save_model()
    # reconize_faces.save_training_labels()
    reconize_faces.open_model()
    reconize_faces.testing_the_trained_model()
    reconize_faces.predict()


