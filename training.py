#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Input, Activation
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.constraints import max_norm
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import time
#from tensorflow.keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import gc
import glob

from newnet.newnet4 import newnet


# In[ ]:



def genmodel(alpha=1, dropout=0.5, learn_rate=0.0001,  epochs=1000, img_width=120, img_height=120):
    
   
    img_rows, img_cols = img_height, img_width
    # number of convolutional filters to use
    #nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    #base_model = newnet(include_top = False, weights='imagenet')
    base_model = newnet(input_shape=(img_height,img_width,3), include_top = False, alpha=alpha, weights=None)
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    #x = Dropout(dropout)(x)

    predictions = Dense(13, activation = 'softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)


    model.summary()
    
    #for layer in base_model.layers:
    #    layer.trainable = False
    # TODO: Compile and train the model here.
    model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=optimizers.Adam(decay=learn_rate/epochs))
    
    return model


# In[ ]:


path ='./onetvt/'
# Initiate the train and test generators with data Augumentation
def trainnet(f=0, alpha = 0.2, dropout = 0.5, learn_rate=0.0001):
    
    train_data_dir = path +"train"
    validation_data_dir =  path +"val"
    train_nums=[]
    test_nums=[]
    img_width, img_height = 120, 120
    batch_size = 128
    epochs = 1000


    for i in range(13):
        tfnum = len(glob.glob(train_data_dir + '/' + str(i) + "/*.png"))
        vfnum = len(glob.glob(validation_data_dir + '/' + str(i) + "/*.png"))
        train_nums.append(tfnum)
        test_nums.append(vfnum)

    nb_train_samples = sum(train_nums)
    nb_validation_samples = sum(test_nums)
    nb_train_max = max(train_nums)
    print(train_nums)
    print("train sample:", nb_train_samples, "validation sample:", nb_validation_samples)

    class_weight = {i: nb_train_max/train_nums[i] for i in range(13)}


    print(class_weight)


    model = genmodel(alpha, dropout, learn_rate, epochs, img_width, img_height)
    if False:
        train_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        fill_mode = "nearest",
        zoom_range = 0.1,
        width_shift_range = 0.2,
        height_shift_range=0.2,
        rotation_range=30)

        test_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        fill_mode = "nearest",
        #zoom_range = 0.3,
        #width_shift_range = 0.3,
        #height_shift_range=0.3,
        #rotation_range=30
        )
    else:
        train_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = "nearest",
        #zoom_range = 0.0,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=180)

        test_datagen = ImageDataGenerator(
        rescale = 1./255)    

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "categorical")

    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")
    mname = "one/" + str(f) + "_ov4nnnewnet_f" + str(alpha) + "_d" + str(dropout) +"_l" + str(learn_rate) +".h5"
    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(mname, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=1, mode='auto')

    
    print(train_generator.class_indices)
    print(validation_generator.class_indices)
    pickle.dump(train_generator.class_indices, open("testnnv2-onenewnetv4.p", "wb" ) )
    
    # Train the model
    history = model.fit_generator(
    train_generator,
    max_queue_size = 48,
    workers = 12,
    use_multiprocessing = False,
    epochs = epochs,
    class_weight=class_weight,
    steps_per_epoch = nb_train_samples/batch_size,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples/batch_size,
    callbacks = [checkpoint, early])
    print(history.history.keys())
    print('train')
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('pn model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('pn model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    del model
    gc.collect()
    time.sleep(60)
    return history.history, mname

    #model.save('my_model.h5')


# In[ ]:



import pickle

if False:
    dropout = [0.2, 0.4, 0.6, 0.8]
    learn_rate = [0.0001, 0.001, 0.01]
    nb_filters = [4, 8, 16]
   
else:
    dropout = [0.5]
    learn_rate = [0.001, 0.01, 0.0001, 1 ,0.1 ]
    alphas = [0.5, 0.8, 1.0]
fmaxaccs =[]
for f in range(10):
    print("Test___________", f, "___________")
    hsts={}
    maxaccs =[]
    for do in dropout:
        for lr in learn_rate:
            for alpha in alphas:
                #print()
                hst,name = trainnet(f, alpha, do, lr)
                hsts[name] = hst
                maxacc = max(hst['val_accuracy'])
                maxaccs.append(maxacc)
                print("OResult:")
                for k, v in hsts.items():
                    print(k + ' acc:' + str(max(v['val_accuracy'])) + ': '+str(f))

    fmaxacc = max(maxaccs)
    fmaxaccs.append(fmaxacc)
    pickle.dump(hsts, open( str(f) + "_newv4fstsave.p", "wb" ) )
    break
    
    
print(*fmaxaccs)    


# In[ ]:



#import gc;
#gc.collect()

def predimg(model, imgs, bs):
    #print(imrs)
    #start_time=time.time()

    preds = model.predict(imgs)

    #print(preds)
    predicted_class = np.argmax(preds, axis=1)
    #print(predicted_class)
    prob = np.empty(bs)
    score1 = np.empty(bs)
    for i in range(bs):
        prob[i] = preds[i][predicted_class[i]]
        score1[i] = preds[i][1]

    #print(prob)
    return predicted_class, prob, score1


# In[ ]:


import glob
X_test = []
y_test = []
for i in range(2):
    data = glob.glob(path + '/test/' + str(i) + '/*.png')
    X_test = X_test + data
    y_test = y_test + [i] * len(data)


# In[ ]:


import numpy as np
import cv2
from numpy import newaxis
import time
bsize = 32

flen = len(X_test)

rlen = flen//bsize
#rlen = 1

rst = flen - rlen * bsize

print('Lens:', flen, 'Blocks:', rlen, 'Bsize:', bsize, 'Rst:', rst)

data = np.zeros((bsize, 120,120,3), dtype=np.float32)

if rst:
    rlen = rlen + 1
y_pred = np.array([])
y_prob = np.array([])
y_score1= np.array([])
for i in range(rlen):
    bs = bsize
    if rst and i == (rlen -1):
        bs = rst
    for j in range(bs):
        fid = i*bsize + j
        image = cv2.imread(X_test[fid])
        imrs = image.astype(float)
        imrs = imrs / 255.0
        data[j,:,:,:] = imrs

    print('i:', i, 'bs', bs)
    ret, prob, score1 = predimg(model, data.copy(), bs)
    
    y_pred = np.hstack((y_pred, ret[:bs]))
    y_prob = np.hstack((y_prob, prob[:bs]))
    y_score1 = np.hstack((y_score1, score1[:bs]))
    #print(ret)
    #print(prob)
    #zzz

print('All:', len(y_pred))
print(y_pred)
print(y_prob)


# In[ ]:


for i in range(len(y_test)):
    print(y_test[i],':',y_pred[i],':' ,y_prob[i])
    if y_test[i] != y_pred[i]:
        print(X_test[i], 'T:',y_test[i], 'P:', y_pred[i])
        image = cv2.imread(X_test[i],0)
        plt.imshow(image, cmap='gray')
        plt.show()


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred,digits =4))
print(confusion_matrix(y_test, y_pred, labels=range(2)))


# In[ ]:


#for i in range(len(y_test)):
#    if y_test[i] != y_pred[i]:
#        print(X_test[i], 'T:',y_test[i], 'P:', y_pred[i])
#        image = cv2.imread(X_test[i],0)
#        plt.imshow(image, cmap='gray')
#        plt.show()
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




