import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
import requests
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf

import h5py, requests, os
import matplotlib.patches as patches

#ZIP_PATH = "https://drive.google.com/drive/folders/1jzHYpTwywUYA53nMGHVROSuVO14hEueq?usp=sharing/"
FILE_NAME ="SynthText_train.h5"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.backend.clear_session()

def sort_points(points, center):
    # calculate the angle of each point from the center point
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    # sort the points by angle
    sorted_points = points[np.argsort(angles)]
    return sorted_points

def draw_points(image, points, color=(255, 0, 0), radius=3):
    # create a copy of the image
    img = image.copy()
    # iterate over the points and draw them on the image
    for point in points:
        cv2.circle(img, tuple(map(int, point)), radius, color, -1)
    return img

def get_bb(img, bbs, indx):
    x1 = int(bbs[0,0,indx])
    y1 = int(bbs[1,0,indx])
    x2 = int(bbs[0,1,indx])
    y2 = int(bbs[1,1,indx])
    x3 = int(bbs[0,2,indx])
    y3 = int(bbs[1,2,indx])
    x4 = int(bbs[0,3,indx])
    y4 = int(bbs[1,3,indx])
    # calculate bounding rectangle
    top_left_x = max(0, min([x1,x2,x3,x4]))
    top_left_y = max(0, min([y1,y2,y3,y4]))
    bot_right_x = max(0, max([x1,x2,x3,x4]))
    bot_right_y = max(0, max([y1,y2,y3,y4]))
    points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    # create an empty image with the same shape as the input image
    mask = np.zeros_like((img))
    # create a list of the bounding box points in the correct format
    bounding_box = np.array([points], dtype=np.int32)
    # fill the area inside the bounding box with white
    cv2.fillPoly(mask, bounding_box, (255, 255, 255))
    # apply the mask to the image
    res = cv2.bitwise_and(img, mask)[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
    #res = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
    #flipping
    """"if(x2 < x1):
        res = cv2.flip(res, 1)
    if(y2 < y1):
        res = cv2.flip(res, 0)
    plt.imshow(res)
    plt.show()"""
    return res

def font_to_num(font):
    if font == 'Alex Brush':
        return 0
    elif font == 'Titillium Web':
        return 1
    elif font == 'Sansation':
        return 2
    elif font == 'Open Sans':
        return 3
    else:
        return 4

def scale_resize_image(image, size):
    image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
    image = tf.image.resize(image, (size, size)) # Resizing the image to 224x224 dimention
    return image


db = h5py.File(FILE_NAME, 'r')
im_names = list(db['data'].keys())
train_x = []
train_y = []
for i in range(0, len(im_names)-1):
    im = im_names[i]
    img  = db['data'][im][:]
    fonts = db['data'][im].attrs['font']
    txts = db['data'][im].attrs['txt']
    charBBs = db['data'][im].attrs['charBB']
    wordBBs = db['data'][im].attrs['wordBB']
    font_indx = 0 
    char_indx = 0
    for j in range(0, len(txts)):
        cropped = get_bb(img, wordBBs, j)
        train_x.append(cropped)
        train_y.append(font_to_num(fonts[font_indx]))
        font_indx += len(txts[j])
        for k in range(0, len(txts[j])):
            cropped = get_bb(img, charBBs, char_indx)
            train_x.append(cropped)
            train_y.append(font_to_num(fonts[char_indx]))
            char_indx+=1
            
for i in range(0, len(train_x)):
    train_x[i] = scale_resize_image(train_x[i], 64)

    #cut the set for train and validation

x_train, x_test, y_train, y_test = train_tet_split(train_x, train_y, test_size=0.2, random_state=42)

print(len(x_train))
print(len(x_test))

Y_train = np_utils.to_categorical(y_train, 5)
Y_test = np_utils.to_categorical(y_test, 5)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='elu', padding='same', input_shape=(64,64,3)))
model.add(Convolution2D(32, 3, 3, activation='elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])

print(len(x_train))
print(len(Y_train))

model = tf.keras.Sequential([
    tf.keras.layers.Convolution2D(32, 3, 3, activation='elu', padding='same',input_shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''
def create_model():
  model=Sequential()
  # Cu Layers 
  model.add(tf.keras.layers.Conv2D(64, kernel_size=(64, 64), activation='relu', input_shape=(64,64,3)))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Conv2D(128, kernel_size=(24, 24), activation='relu'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Conv2DTranspose(128, (24,24), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
  model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
  model.add(tf.keras.layers.Conv2DTranspose(64, (12,12), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
  model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
  #Cs Layers
  model.add(tf.keras.layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))
  model.add(tf.keras.layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))
  model.add(tf.keras.layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))
  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2383,activation='relu'))
  model.add(Dense(5, activation='softmax'))
  return model

model= create_model()'''
X_train = np.array(x_train)
Y_train = np.array(Y_train)
#X_train = X_train.reshape(X_train.shape[0], 1, 64, 64)
#X_train = X_train.astype('float32')
#X_train /= 255
print(X_train.shape)
print(Y_train.shape)
plt.imshow(X_train[0])
plt.show()

model.fit(X_train, Y_train, batch_size=16, epochs=10, verbose=1)

X_test = np.array(x_test)
np.random.shuffle(Y_test)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

print(test_acc)