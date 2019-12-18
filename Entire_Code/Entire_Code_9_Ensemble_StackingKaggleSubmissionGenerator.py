#Predict on the test set (unseen data) test_images data


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

# ENSEMBLE PREDICTIONS AND SUBMIT
results = np.zeros( (Test_Set.shape[0],10) ) 
for j in range(nets):
    results = results + model[j].predict(Test_Set)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Category")

#save in csv file
submission = pd.concat([pd.Series(range(0,10000),name = "Id"),results],axis = 1)
submission.to_csv('../root/COMP 551-Project 3-Data/EnsembleCNN_AugmentedData_Epochs10_Batch64.csv',index=False)

