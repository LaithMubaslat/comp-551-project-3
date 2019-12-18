#Larger Convolutional Neural Network

#We import classes and function then load and prepare the data the same as in the previous CNN example.
# Larger CNN for the MNIST Dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
# fix random seed for reproducibility

# fix random seed for reproducibility





# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define the larger model
def larger_model():
	# create model
	model2 = Sequential()
	model2.add(Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='relu'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Conv2D(15, (3, 3), activation='relu'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Dropout(0.2))
	model2.add(Flatten())
	model2.add(Dense(128, activation='relu'))
	model2.add(Dense(50, activation='relu'))
	model2.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model2

#Like the previous two experiments, the model is fit over 10 epochs with a batch size of 200.
# build the model
model2 = larger_model()
# Fit the model
model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model2.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))




#Results
#10 epochs: 0.9315 accuracy (the best on kaggle so far)
#30 epochs: 0.93225
#100 epochs: 0.93 