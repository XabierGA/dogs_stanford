import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D , Flatten , Dropout , MaxPooling2D , BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split


class CNN_dogs():

    def __init__(self , batch , epochs , IMG_shuff , Y_shuff , num_classes):

        self.batch_size = batch
        self.epochs = epochs
        self.X_train ,self.X_test , self.y_train , self.y_test = train_test_split(IMG_shuff , Y_shuff)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        self.num_classes = num_classes
        self.model = Sequential()

    def add_layers(self):

        self.model.add(Conv2D(32 , (3,3) , activation = 'relu', input_shape=(pixels,pixels,rgb)))
        self.model.add(Conv2D(32 , (3,3) , activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(BatchNormalization(axis = -1))

        self.model.add(Conv2D(64 , (3,3) , activation ='relu'))
        self.model.add(Conv2D(64 , (3,3) , activation ='relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(BatchNormalization(axis = -1))

        self.model.add(Conv2D(128 , (3,3) , activation = 'relu'))
        self.model.add(Conv2D(128 , (3,3) , activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(BatchNormalization(axis = -1))

        self.model.add(Conv2D(256 , (3,3) , activation = 'relu'))
        self.model.add(Conv2D(256 , (3,3) , activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(BatchNormalization(axis = -1))


        self.model.add(Flatten())
        self.model.add(Dense(500, activation='relu'))
        self.model.add(BatchNormalization(axis = -1))
        self.model.add(Dropout(.25))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(BatchNormalization(axis = -1))
        self.model.add(Dropout(.125))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(BatchNormalization(axis = -1))

        self.model.add(Dense(self.num_classes , activation = "softmax"))


    def compile_model(self , verbose = 1):

        self.model.compile(loss = 'categorical_crossentropy' , optimizer = Adadelta() , metrics = ['accuracy'])
        self.model.fit(self.X_train , self.y_train , batch_size = self.batch_size , epochs = self.epochs , verbose = verbose , validation_data = (self.X_test , self.y_test))

    def score(self , verbose = 0):

        score = self.model.evaluate(self.X_test , self.y_test , verbose = verbose)
        print("Test loss: " , score[0])
        print("Test accuracy: " , score[1])
