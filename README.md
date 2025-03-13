# Arch-I-Scan Project Sherd Classification Model and Code Repository
This repository contains code and Convolutional Neural Network (CNN) models coming out of the Arch-I-Scan Project, the work of which towards automatic classification of sherd material is published as ***INTERNET ARCHAEOLOGY REFERENCE***.
Repository contains:
1.	*CNN models* developed to automatically classify sherds of Roman-era terra sigillata pottery, produced in Gaul, from archaeological contexts in Britain can be found in Releases. Derscrioption of models is presented below.
2.	*Simulation code* is the code that was used to generate the images of simulated sherds for pretraining the CNN model in order to compensate for the limited size of the dataset of photographs of real sherds.
3.	*CNN training code* is the code that was used to train CNN.
The photograph data used to train these models, as well as the 3D models of vessel forms that underlie the simulation process can be found at (***REF ADS***).

## CNN Models
Models can be found in Releases. 
This repository contains the formed CNN models developed to automatically classify sherds of Roman-era terra sigillata pottery, produced in Gaul, from archaeological contexts in Britain. Two models, representing two stages of the training process, are made available: **OnSimulData** and **OnRealData**. Both models have the same architecture:
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
Input size of image is `img_height=img_width=224`
All layers are trainable.
After convolutional pare of InceptionV3 we added four layers: Dropout(0.50), GlobalAveragePooling2D(), Dense(512, activation='relu'), and Dense(NUM_CLASSES, activation='softmax').
`NUM_CLASSES=5`.

**OnSimulData** is trained on simulated data only: Inception V3 network, trained on simulated sherds, the sherds were sized 10-30% of the whole vessels, the model was trained for 10 epochs before overfitting.

**OnRealData** is **OnSimulData** additionally trained on real data: Inception V3 network, fine-tuned on training set of 5 real classes (25 epochs)

To access models you should click word "Releases" and then select one of two models.


## Simulation code
Code to simulate artificial sherds can be found in folder **Simulations**. Instruction is presented in file **Simulations/ReadMe.docx**

## CNN training code
Code used for training of CNN can be found in file **InceptionV3_training_evaluation.py**

## Acknowledgment
The authors would like to thank the Arts and Humanities Research Council (UK) for funding the Arch-I-Scan Project (Grant number AH/T001003/1) and the partner organisations for access to their terra sigillata collections.

## Citation of Arch-I-Scan project data
Please cite this dataset as<br>
Núñez Jareño, S.J.; van Helden, D.P.; Mirkes, E.M.; Zheng Z.; Tyukina, T.A.; Tyukin, I.Y.; Allison, P. Arch-I-Scan data repository. Available online https://github.com/ArchiScn/Access, 2025.
