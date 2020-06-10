# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:55:53 2019

@author: Shrikrishna
"""


import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt



def extractFace(grayImg):
    faces = faceCascade.detectMultiScale(grayImg, 1.3, 5)
    (x,y,w,h) = faces[0]
    return grayImg[y:y+w, x:x+h], faces[0]


def dataPrep(dirs,folder):
    faces = []
    labels = []
    for dirName in dirs:
        label = dirName
        personPath = folder + dirName
        personImgs = os.listdir(personPath)
        for imgs in personImgs:
            imgPath = personPath + "/" + imgs
            image = cv2.imread(imgPath)
            grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(imgPath)
            face, rect = extractFace(grayImg)
            face=cv2.resize(face,(250,250))
            faces.append(face)
            labels.append(label)
    return faces,labels
            

def classify(X_train_pca,X_test_pca,test_x,train_y,test_y):
    
    ###########
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
    clf = clf.fit(X_train_pca, train_y)
    svmpred = clf.predict(X_test_pca)
    prediction_titles = [title(svmpred, test_y, train_y, i) for i in range(svmpred.shape[0])]
    print("Classification Method: Linear SVM")
    plotImages(test_x, prediction_titles, 250, 250)


    
    ###########
    nbmodel = GaussianNB()
    nbmodel.fit(X_train_pca,train_y)
    nbpred = nbmodel.predict(X_test_pca)
    prediction_titles = [title(nbpred, test_y, train_y, i) for i in range(nbpred.shape[0])]
    print("Classification Method: Naive Bayes")
    plotImages(test_x, prediction_titles, 250, 250)
    
    
    ###########
    mlpclf = MLPClassifier()
    mlpclf.fit(X_train_pca,train_y)
    mlpPred = mlpclf.predict(X_test_pca)
    prediction_titles = [title(mlpPred, test_y, train_y, i) for i in range(mlpPred.shape[0])]
    print("Classification Method: Multilayer Perceptron")
    plotImages(test_x, prediction_titles, 250, 250)

    
    ##########
    gbm = GradientBoostingClassifier(n_estimators=1000)
    gbm.fit(X_train_pca,train_y)
    gbmpred = gbm.predict(X_test_pca)
    prediction_titles = [title(gbmpred, test_y, train_y, i) for i in range(gbmpred.shape[0])]
    print("Classification Method: Gradient Boosting")
    plotImages(test_x, prediction_titles, 250, 250)

    
 
def plotImages(images,titles, h, w, n_row=2, n_col=5):

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())    

def title(y_pred, y_test, target_names, i):
    
    pred_name = y_pred[i]
    true_name = y_test[i]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
        
    
        
def main():
    faceCascade = cv2.CascadeClassifier('D:\\Sem 2\\CV\\haarcascade_frontalface_default.xml')
    trainImageFolder = "D:/Sem 2/CV/ImageDataset/"
    trainDirs  = os.listdir(trainImageFolder)
    train_x,train_y = dataPrep(trainDirs,trainImageFolder)
    testImageFolder = "D:/Sem 2/CV/imgTest/"
    testDirs = os.listdir(testImageFolder)
    test_x,test_y = dataPrep(testDirs,testImageFolder)
    
    train_x = np.stack(train_x)
    train_x = train_x.transpose(1,2,0).reshape(-1,train_x.shape[0])
    train_x = train_x.transpose(1,0)
    
    test_x = np.stack(test_x)
    test_x = test_x.transpose(1,2,0).reshape(-1,test_x.shape[0])
    test_x = test_x.transpose(1,0)


    pca = PCA(n_components=10, svd_solver='randomized',
          whiten=True).fit(train_x)
    eigenfaces = pca.components_.reshape((10, 250, 250))
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plotImages(eigenfaces,eigenface_titles,250,250,5,4)

    X_train_pca = pca.transform(train_x)
    X_test_pca = pca.transform(test_x)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }


    classify(X_train_pca,X_test_pca,test_x,train_y,test_y)






