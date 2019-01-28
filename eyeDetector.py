import cv2
import sys
import json
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
#from sklearn.cross_validation import train
from sklearn.model_selection import train_test_split

import struct
import numpy as np
import cv2


from PIL.ImageWin import Window


def posReader(name):
    with open(name) as f:
        data = json.load(f)
    return data["eye_details"]["look_vec"]

    co = [0.3914, 0.0591, -0.9183,1]
    hsc = co * object.getMatrix() * Window.GetPerspMatrix()

    # now you have unnormalized (homogenious) window coordinates
    # the scaling factor is in the w-component (index 3) -&gt; divide x,y by w (if you need z also to compare against a z-buffer value, do the same for hsc[2]

    hsc[0] /= hsc[3]
    hsc[1] /= hsc[3]

    # now hsc[0] is the normalized x-coordinate, hsc[1] is the normalized y-coordinate (in the range [-1..1] ,  0,0 corresponds to the center of the window
    # next convert to screen coordinates i.e multiply by half the window size and add to window center

    w = Window.GetScreenInfo(Window.Types.VIEW3D)[0]["vertices"]  # the window vertices
    sx = w[2] - w[0]  # size_x
    sy = w[3] - w[1]  # size_y
    mx = 0.5 * (w[0] + w[2])  # center_x
    my = 0.5 * (w[1] + w[3])  # center_y

    x = mx + hsc[0] * 0.5 * sx  # center_x + hsc_x  * half size_x
    y = my + hsc[1] * 0.5 * sy  # center_y + hsc_y * half size_y

    # here we are - screen coordinates x,y
    print(co)
    print(x,y)

import numpy as np
def LeNet(train_data, train_labels,test_data, test_labels,inps,d):#(32,32,1):
    model = Sequential()
    #1
    model.add(Convolution2D(
        filters=32,
        kernel_size=(7, 7),
        padding="same",
        strides=(1, 1),
        input_shape=inps))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))
    #2
    model.add(Convolution2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same"))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))
    # 2

    model.add(Convolution2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same"))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))

    model.add(Flatten())
    #3
    #model.add(Dense(128))

    #model.add(Activation(
    #    activation="relu"))
    #4
    #model.add(Dense(128))

    #model.add(Activation(
    #    activation="relu"))
    #5
    model.add(Dense(d))

    model.add(Activation("softmax"))

    model.compile(
        #loss="categorical_crossentropy",
        #optimizer=SGD(lr=0.01),
        loss='mean_absolute_error',#rmsprop
        optimizer='adam',
        metrics=["accuracy"])

    model.fit(
        train_data,
        train_labels,
        batch_size=128,
        nb_epoch=10)

    (loss, accuracy) = model.evaluate(
        test_data,
        test_labels,
        batch_size=128,
        verbose=1
    )
    model.save('my_model_epoch_10.h5')
    return loss,accuracy

def img_getter(add,size,weight, height):
    out = []
    for i in range(size):
        img = cv2.imread('eyes/'+str(i+1)+add)
        #print('--------')
        if img is None:
            img = cv2.imread('imgs/' + str(i + 1) + '.jpg')
            mean,std = cv2.meanStdDev(img, mask=None)
            img[:,:,0] = (img[:,:,0] -mean[0])/std[0]
            img[:,:,1] = (img[:,:,1] -mean[1])/std[1]
            img[:,:,2] = (img[:,:,2] -mean[2])/std[2]
            reshapedimage = cv2.resize(img, (weight, height))
            out.append(reshapedimage)
            continue
        reshapedimage = cv2.resize(img, (weight, height))
        out.append(reshapedimage)
    return out

def dataset_seperator(img,label,size):
    dist = []
    for i in range(size):
        d = np.linalg.norm(label[i] - label[:, None], axis=-1)
        dist.append(d)
    mean = np.mean(dist)
    max = np.mubuax(dist)
    thresh = (mean+max)/2
    classes = []
    cord = []
    for i in range(size):
        if dist[i]>thresh:
            classes.append(img[i])
            cord.append(label[i])
    print(len(classes))
    label2 = []
    for i in range(len(img)):
        d = np.linalg.norm(label[i] - cord[:, None], axis=-1)
        index = np.argmin(d)
        label2.append(index)
    X_train, X_test, y_train, y_test = train_test_split(img, label2, test_size=0.2)
    return X_train, X_test, y_train, y_test



out = np.zeros([42080, 3])
def label_reader(add,size,x):

    pos = []
    for i in  range(size):
        pos.append(posReader(add+'/'+ str(i+1) + '.json').replace('(', '').replace(')', '').split(","))
    for i  in range(len(pos)):
        out[i+x][0] = float(pos[i][0])
        out[i+x][1] = float(pos[i][1])
        out[i+x][2] = float(pos[i][2])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
x = np.array([np.array([1,2,3,4]),np.array([11,12,13,14]),np.array([21,22,23,24])])
y = x.flatten()
y = y.reshape(3,4)
print(y)
#\label, img = dataset_eye_extractor(130,100,100)
img1 = img_getter('-10.jpg',10267,100,100)
img2 = img_getter('-20.jpg',10203,100,100)
img3 = img_getter('-30.jpg',21610,100,100)
img =img1+img2+img3
l1 = label_reader('imgs-10',10267,0)
l2 = label_reader('imgs-20',10203,10267)
l3 = label_reader('imgs',21610,20470)
label = out
print(np.shape(img))
print(np.shape(label))
dataset_seperator(img,label,42080)
train,test,train_l,test_l= dataset_seperator(img,label,42080)
#for i in range(len(img)):
#    if i % 10 == 0:
##        test.append(img[i])
 #       test_l.append(label[i])
 #   else:
 #       train.append(img[i])
 #       train_l.append(label[i])
print(np.shape(train))
print(np.shape(test))
train_l = np.array(train_l).flatten()
train_l = train_l.reshape(18423, 3)
test_l = np.array(test_l).flatten()
test_l = test_l.reshape(2047, 3)
print(np.shape(train_l))
print(np.shape(test_l))
#print(LeNet(np.array(train), np.array(train_l), np.array(test), np.array(test_l), (100, 100, 3), 3))
