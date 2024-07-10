# Sherd project
This repository contains code and models for ArchIScan project for sherd recognition.

## Models
Models can be found in Releases. 
We publish two models: **OnSimulData** and **OnRealData**. Both models have the same architecture:

```
net =InceptionV3(weights='imagenet', include_top=False,  input_shape=(img_height,img_width,3))

  net.trainable = True# all layers trainable

  model = Sequential([  # extension
       net,
        # here is our custom prediction layers
       Dropout(0.50),       
       GlobalAveragePooling2D(),
       Dense(512, activation='relu'),  
       Dense(NUM_CLASSES, activation='softmax')
    ])
```
We used architecture InceptionV3 without fully connected layers.
Input size of image is img_height=??? and img_width=???
All layers are trainable.
After convolutional pare of InceptionV3 we added four layers: Dropout(0.50), GlobalAveragePooling2D(), Dense(512, activation='relu'), and Dense(NUM_CLASSES, activation='softmax').
NUM_CLASSES

**OnSimulData** is trained on simulated data only: Inception V3 network, trained on simulated sherds, sizes of the sherds 10-30% of the whole vessels, trained for 10 epochs before overfitting.

**OnRealData** is **OnSimulData** additionally trained on real data: Inception V3 network, fine-tuned on training set of 5 real classes (25 epochs)

To access models you should click word "Releases" and then select one of two models.



Code for training of CNN can be found in  ???

Code to simulate artificial sherds can be found in ???
