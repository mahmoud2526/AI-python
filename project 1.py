import cv2
import os
import glob
import numpy as np
import pandas as pd
from skimage.feature import  graycomatrix,graycoprops
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,precision_score,f1_score,recall_score
#simple classifer sticaking
size=128
all_images=[]
all_labels=[]
for path in glob.glob(r"C:\Users\Eng mahmoud\OneDrive\Desktop\iot course (1)\PlantVillage\potato\*"):
    lable=path.split("\\")[-1]
    print(lable)
    #glob.glob بدخلك الفلولدر
    #os.path.join() للبحث في الداخل تحتاج عنوان url و نوع الفايللاjpg
    for img_path in glob.glob(os.path.join(path,'*.jpg')):
        img=cv2.imread(img_path,0)
        img=cv2.resize(img,(size,size))
        all_images.append(img)
        all_labels.append(lable)

    print("ok")

all_images=np.array(all_images)
all_labels=np.array(all_labels)
print("ok2",len(all_images))
#print(all_images[10][10][10])
#glcm it takes angles 0 mean he will search pattern in horzintal and 90 vertical
xtrain,xtest,ytrain,ytest=train_test_split(all_images,all_labels,test_size=0.3,random_state=42)
le=LabelEncoder()
y_train=le.fit_transform(ytrain)
y_test=le.fit_transform(ytest)


def glcm(image_data):
    feature=pd.DataFrame()

    for img in range(image_data.shape[0]):

        df=pd.DataFrame()

        img=image_data[img,:,:]
        distance1=[1,5,2]
        angles=[0,np.pi/2,np.pi/4]
        img_dataframe=pd.DataFrame()
        for i in range(len(distance1)):
            for j in range(len(angles)):
                Glcm=graycomatrix(img,[distance1[i]],[angles[j]])
                df['Energy']=graycoprops(Glcm,"energy")[0]
                df["corrleation"]=graycoprops(Glcm,"correlation")[0]
                df["diss_sim"]=graycoprops(Glcm,"dissimilarity")[0]
                df["Homogenous"]=graycoprops(Glcm,"homogeneity")[0]
                df["contrast"]=graycoprops(Glcm,"contrast")[0]
                #img_dataframe=img_dataframe._append(df)
        feature=feature._append(df)
        #print(img_dataframe)
    return feature

#hog ,sift,harris computer vision
train=glcm(xtrain)
print('ok3')
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
DT=DecisionTreeClassifier()
DT.fit(train,y_train)
test=glcm(xtest)
x_input=np.reshape(test,(xtest.shape[0],-1))
print(DT.score(x_input,y_test))
svc=SVC()
svc.fit(train,y_train)
print(svc.score(x_input,y_test))
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(n_neighbors=)



