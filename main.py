# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:23:39 2018

@author: John
"""
#imports 
from bs4 import BeautifulSoup
import re
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#methods
def main():
    print("Begin")
    trainX, trainY, testX, testY = readFiles()   
    
    print("\n\n")
    print("Train X,Y: " , len(trainX), len(trainY))
    print("Test X,Y: " , len(testX), len(testX))
    
    X_train = preProcess(trainX)
    X_test = preProcess(testX)
    y_train = trainY
    y_test = testY
    
    #classification
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
    classifier.fit(X_train, y_train)  
    
    #prediction
    y_pred = classifier.predict(X_test)  
    
    #evaluation
    print("\nRandom Forest\n")
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))  
    print("Classification report:\n", classification_report(y_test,y_pred))  
    print("Accuracy:\n", accuracy_score(y_test, y_pred))  
    
    #kNN
    from sklearn.neighbors import KNeighborsClassifier  
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(X_train, y_train)  
    
    #prediction
    y_pred = classifier.predict(X_test)  
    
    #evaluation
    print("\nkNN\n")
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))  
    print("Classification report:\n", classification_report(y_test,y_pred))  
    print("Accuracy:\n", accuracy_score(y_test, y_pred))  
    
    #SVM
    from sklearn import svm 
    classifier = svm.SVC(kernel = "linear")  
    classifier.fit(X_train, y_train)  
    
    #prediction
    y_pred = classifier.predict(X_test)  
    
    #evaluation
    print("\nSVM\n")
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))  
    print("Classification report:\n", classification_report(y_test,y_pred))  
    print("Accuracy:\n", accuracy_score(y_test, y_pred))  
    
    
    #SVM
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)  
    
    #prediction
    y_pred = classifier.predict(X_test)  
    
    #evaluation
    print("\nNaive Bayes\n")
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))  
    print("Classification report:\n", classification_report(y_test,y_pred))  
    print("Accuracy:\n", accuracy_score(y_test, y_pred)) 
    
    
    #end of main
    
def preProcess(X):
    #pre-processing
    lemmatizer = WordNetLemmatizer()
    documents = []
    for sen in range(0, len(X)):  
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
    
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    
        # Converting to Lowercase
        document = document.lower()
    
        # Lemmatization
        document = document.split()
    
        document = [lemmatizer.lemmatize(word) for word in document]
        document = ' '.join(document)
    
        documents.append(document)
        
    #converting text to numbers
    from sklearn.feature_extraction.text import CountVectorizer 
    #change the parameters of the following method
    vectorizer = CountVectorizer(max_features=250, min_df=0.01, max_df=0.15, stop_words=stopwords.words('english'))  
    X = vectorizer.fit_transform(documents).toarray() 
    
    #TFIDF
    from sklearn.feature_extraction.text import TfidfTransformer  
    tfidfconverter = TfidfTransformer()  
    X = tfidfconverter.fit_transform(X).toarray()  
    
    return X
    #end of preProcess

def parseWords(path):
    contents = open(path, errors='ignore')
    soup = BeautifulSoup(contents, "lxml")
    all_text = ''.join(soup.get_text())        
    all_text = re.sub('\n', ' ', all_text)  #remove new lines
    all_text = re.sub(' +',' ', all_text)   #remove multiple white spaces
    return all_text
    #end of parseWords
    
def readFiles():
    root= "webkb"
    targetLabels = os.listdir(root)    
    
    trainXfiles = []
    trainX = []
    trainY = []
    testXfiles = []
    testX = []
    testY = []
    
    for label in targetLabels:
        path = root + "/" + label
        universities = os.listdir(path)
        for university in universities:
            count = 0
            subpath = path + "/" + university
            files = os.listdir(subpath)
            for file in files:
                filePath = subpath+"/"+file 
                #take wisconsin as test data
                if(university == "wisconsin"): 
                    testY.append(label)
                    testXfiles.append(filePath)
                    testX.append(parseWords(filePath))
                else:
                    trainY.append(label)
                    trainXfiles.append(filePath)
                    trainX.append(parseWords(filePath))
                #limiting dataset; delete lines unitil return
#                print(label, university, count)
#                count += 1
#                if(count>20): 
#                    break
#                break #limit to 1 file
    
    return trainX, trainY, testX, testY
    #end of readFiles
    
#execution
main()
 