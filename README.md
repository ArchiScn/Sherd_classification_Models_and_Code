# Arch-I-Scan Project Sherd Classification Model and Code Repository
This repository contains the Convolutional Neural Network (CNN) models and code used and developed by the Arch-I-Scan Project, towards automatic classification of sherd material is published as ***INTERNET ARCHAEOLOGY REFERENCE***.
Repository contains:
1.	*CNN models* developed to automatically classify sherds of Roman-era *terra sigillata* pottery, produced in Gaul, from archaeological contexts in Britain can be found in Releases. Description of the models is presented below.
2.	*Simulation code* is the code that was used to generate the images of simulated sherds for pretraining the CNN model in order to compensate for the limited size of the dataset of photographs of real sherds.
3.	*CNN training code* is the code that was used to train the CNN models.
The photograph data used to train these models, as well as the 3D models of vessel forms that underlie the simulation process can be found at (***REF ADS***).

## CNN Models
The models can be found in **Releases**. 
This repository contains the formed CNN models developed to automatically classify sherds of Roman-era *terra sigillata* pottery, produced in Gaul, from archaeological contexts in Britain. Two models, representing two stages of the training process, are made available: **OnSimulData** and **OnRealData**. Both models have the same architecture:
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
The input size of image is `img_height=img_width=224`
All layers are trainable.
After convolutional part of InceptionV3 we added four layers: Dropout(0.50), GlobalAveragePooling2D(), Dense(512, activation='relu'), and Dense(NUM_CLASSES, activation='softmax').
`NUM_CLASSES=5`.

**OnSimulData** is trained on simulated data only: Inception V3 network, trained on simulated sherds, the sherds were sized 10-30% of the whole vessels, the model was trained for 10 epochs before overfitting.

**OnRealData** is **OnSimulData** additionally trained on real data: Inception V3 network, fine-tuned on training set of 5 real classes (25 epochs)

To access the models you should click the menu item 'Releases" and then select one of the two models.


## Simulation code
The code to simulate artificial sherds can be found in the folder **Simulations**. Instructions are presented in file **Simulations/ReadMe.docx**

## CNN training code
Code used for training of CNN can be found in file **InceptionV3_training_evaluation.py**

## Acknowledgements
The authors would like to thank the Arts and Humanities Research Council (UK) for funding the Arch-I-Scan Project (Grant number AH/T001003/1) and the project’s partner organisations – MOLA, the Vindolanda Trust, University of Leicester Archaeology Service, and the Colchester and Ipswich Museum Service – for access to their Roman ceramics collections.

## Citation for the Arch-I-Scan Project Repositories
Please cite this repository as<br>
Mirkes, E.M.; van Helden, D.P.;; Zheng Z.; Tyukina, T.A.; Tyukin, I.Y.; Núñez Jareño, S.J.; Allison, P. The Arch-I-Scan Project repositories. Available online https://github.com/ArchiScn/Access, 2025.
