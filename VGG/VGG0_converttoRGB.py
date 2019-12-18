
SIZE=64

dim = (SIZE, SIZE)
X_train=X_train.astype(float)/255

#Y_train=train_y
X_valid=X_test.astype(float)/255
#Y_valid=valid_y

#convert 28x28 grayscale to 48x48 rgb channels
def to_rgb(img):
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    img_rgb=np.swapaxes(img_rgb, 1, 2)
    img_rgb=np.swapaxes(img_rgb, 0, 1)
    return img_rgb

rgb_list = []
rgbval_list=[]
#convert X_train data to 48x48 rgb values
for i in range(len(X_train)):
    rgb = to_rgb(X_train[i])
    #rgbval=to_rgb(X_valid[i])
    rgb_list.append(rgb)
    #rgbval_list.append(rgbval)
    #print(rgb.shape)
    
for i in range(len(X_valid)):
    #rgb = to_rgb(X_train[i])
    rgbval=to_rgb(X_valid[i])
    #rgb_list.append(rgb)
    rgbval_list.append(rgbval)
    #print(rgb.shape)
    

    
rgb_arr = np.stack([rgb_list],axis=4)
rgb_arr_to_3d = np.squeeze(rgb_arr, axis=4)



rgb_arrval=np.stack([rgbval_list],axis=4)
rgb_arr_to_3dval = np.squeeze(rgb_arrval, axis=4)


#X_train=rgb_arr_to_3d
#X_valid=rgb_arr_to_3dval

print(rgb_arr_to_3d.shape)