#Predict on the test set (unseen data) test_images data


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Preprocess test set
img=test_images.copy()

  


largest= largest_digit_set(img)

# reshape to be [samples][pixels][width][height]
Test_Set = largest.reshape(largest.shape[0], 1, 64, 64).astype('float32')

# normalize inputs from 0-255 to 0-1
Test_Set = Test_Set / 255
# one hot encode outputs
#y_test = np_utils.to_categorical(Test_Set)
#num_classes = y_test.shape[1]

#Predict
prediction=model.predict(Test_Set)
#prediction
#plt.imshow(test_images[20])
#prediction_file = pd.DataFrame(prediction, columns=['prediction']).to_csv('prediction.csv')


# predict results
results = model.predict(Test_Set)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Category")

#plt.imshow(test_images[0])



#save in csv file
submission = pd.concat([pd.Series(range(0,10000),name = "Id"),results],axis = 1)
submission.to_csv('../root/COMP 551-Project 3-Data/SimpleCNN_Model.csv',index=False)

