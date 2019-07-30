# Importing the Keras libraries and packages
from keras.models import Sequential, Model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# We will use dropout to discard neurons during the learning process, which helps preventing overfitting
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Importing libraries and packages
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image
from numpy import savetxt
from math import ceil



# preparing the test dataset as Keras requirement

unready_dir = os.path.join(os.getcwd(), 'sc5-2013-Mar-Apr-Test-20130412/')
ground_truth_file = os.path.join(unready_dir, 'ground_truth.txt')

if not os.path.exists('valdation_data'):
        os.mkdir('valdation_data')
else:
        print('Folder exists')


valdation_path = os.path.join(os.getcwd(), 'valdation_data/')
# print(valdation_path)


test_dataset = pd.read_csv(ground_truth_file, delimiter=';')
test_dataset.columns = ['image_names', 'categories']
file_names = list(test_dataset['image_names'])
img_labels = list(test_dataset['categories'])


folders_to_be_created = np.unique(list(test_dataset['categories']))

# print(folders_to_be_created)



for new_path in folders_to_be_created:
        if not os.path.exists(os.path.join(valdation_path, new_path)):
                if new_path[0:8] == "Snapshot":
                        if not os.path.exists(os.path.join(valdation_path, 'water')):
                                os.mkdir(os.path.join(valdation_path, 'water'))
                        else:
                                continue
                else:
                        os.makedirs(os.path.join(valdation_path, new_path))
        else:
                print('Folder exists')
                


folders = folders_to_be_created.copy()

# print(folders)

for f in range(len(file_names)):
        current_img = file_names[f]
        current_label = img_labels[f]
        ##**Check this Line Accordingly**
        try:
                if current_label[0:8] =="Snapshot":
                        shutil.move(unready_dir+current_img, valdation_path+'water')
                else:
                        shutil.move(unready_dir+current_img, valdation_path+current_label)
        except:
                print("Image exists")

sizeOfTraining_folder = len(next(os.walk('sc5'))[1])
sizeOfTest_folder = len(next(os.walk('valdation_data'))[1])

if (sizeOfTest_folder < sizeOfTraining_folder): 
        for x in range(sizeOfTest_folder, sizeOfTraining_folder):
                os.mkdir(os.path.join(valdation_path, 'empty_folder'+str(x)))
else:
        print('all set')


# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(activation = "relu", units = 128))
classifier.add(Dropout(0.1))
classifier.add(Dense(activation = "relu", units = 128))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation = "relu", units = 128))
classifier.add(Dropout(0.5))


# Adding the output layer¶ 
classifier.add(Dense(activation = "softmax", units = 20))


print('\nBefore model compiling: \n')
print(classifier.summary())

# Compiling the CNN
classifier.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])

print('\nAfter model compiling: \n')
print(classifier.summary())

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'sc5/',
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical")

test_set = test_datagen.flow_from_directory(
        'valdation_data/',
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical")

# print(test_set)
history = classifier.fit_generator(
        training_set,
        steps_per_epoch=ceil(4799/32),
        epochs=20,
        validation_data=test_set,
        shuffle=True,
        validation_steps=ceil(1968/32))

# Saving the weights
classifier.save_weights('second_try.h5')


score= classifier.evaluate_generator(test_set, steps=ceil(1968/32), verbose=0)
print('Test loss:', score[0])
print('Test Accuracy:', score[1]*100)

classifier.predict_generator(test_set, steps=ceil(1968/32), verbose=0)

# plt.figure(figsize = (18,8))
# plt.subplot(121)
# plt.plot(history.history['loss'], color='darkcyan', label="Training loss")
# plt.plot(history.history['val_loss'], color='maroon', label="validation loss",)
# plt.xlabel('Epochs')
# plt.ylabel('loss')
# plt.legend(loc='best', shadow=True)
# plt.title('4-Convolution Layer Loss')
# plt.subplot(122)
# plt.plot(history.history['acc'], color='darkcyan', label="Training accuracy")
# plt.plot(history.history['val_acc'], color='maroon',label="Validation accuracy")
# plt.xlabel('Epochs')
# plt.ylabel('accuracy')
# plt.legend(loc='best', shadow=True)
# plt.title('4-Convolution Layer Accuracy')
# plt.show()

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.style.use("ggplot")
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




# Part 3 - Making new predictions

test_image_raw = "predictions/p7.jpeg"
test_image = image.load_img(test_image_raw, target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
resultDic = training_set.class_indices
# print(result)
# print(training_set.class_indices)

img_class = classifier.predict_classes(test_image)
# print(img_class)
classname = img_class[0]
prediction =  list(resultDic.keys())[list(resultDic.values()).index(img_class)]
print("Category: ",classname)
print("The boat belong to: ",prediction)

Image(filename = test_image_raw)

img = mpimg.imread(test_image_raw)
imgplot = plt.imshow(img)
plt.title(classname)
plt.show()




# -------------------

# img = image.load_img(path="predictions/new.jpg",color_mode="grayscale",target_size=(128,128,1))
# img = image.img_to_array(img)
# test_img = img.reshape((1,16384))

# result = classifier.predict(test_img)
# resultDic = training_set.class_indices

# img_class = classifier.predict_classes(test_img)

# classname = img_class[0]

# prediction =  list(resultDic.keys())[list(resultDic.values()).index(img_class)]
# print("Category: ",classname)
# print("The boat belong to: ",prediction)

# img = img.reshape((128,128))
# plt.imshow(img)
# plt.title(classname)
# plt.show()