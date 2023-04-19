#Import required library
print("# loading pandas, numpy, and matplotlib")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("# loading sklearn")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
print("# loading imblearn")
from imblearn.over_sampling import SMOTE
#print("# loading keras")
#from keras.models import Sequential
#from keras.layers import Dense

print("Done importing libraries")

print("Reading genome csv file")
gen = pd.read_csv("gen_mean_step_diff.csv")
gen["step_diff"].fillna(0, inplace=True)
gen["step_diff_30min"].fillna(0, inplace=True)
gen["step_diff_hr"].fillna(0, inplace=True)
gen["step_diff_one_half"].fillna(0, inplace=True)
gen["step_diff_3hr"].fillna(0, inplace=True)
gen["step_diff_6hr"].fillna(0, inplace=True)
gen["step_diff_12hr"].fillna(0, inplace=True)
print("Done reading")

print("PRINCIPAL COMPONENT ANALYSIS")
X = gen.loc[:, ~gen.columns.isin(['Time', 'AnimalID', 'dt', 'key', 'estrus'])]
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X)

print("USING 6 COMPONENTS")
pca_1 = PCA(n_components=6)
X_pca_1 = pca.fit_transform(X)

print("---BALANCING DATA---")
y = gen['estrus'].values
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_pca_1, y, test_size=0.4, random_state=0)

# resample on training data
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
# balanced train data to be used
os_data_X = pd.DataFrame(data=os_data_X)
os_data_y= pd.DataFrame(data=os_data_y,columns=['estrus'])

# check the numbers of the data
print("Length of oversampled data is ",len(os_data_X))
print("Number of not estrus in oversampled data",len(os_data_y[os_data_y['estrus']==0]))
print("Number of estrus",len(os_data_y[os_data_y['estrus']==1]))
print("Proportion of not estrus data in oversampled data is ",len(os_data_y[os_data_y['estrus']==0])/len(os_data_X))
print("Proportion of estrus data in oversampled data is ",len(os_data_y[os_data_y['estrus']==1])/len(os_data_X))


print("---CLASSIFICATION TREE---")
tree_model = DecisionTreeClassifier()
# train the classifier on the training data
tree_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
tree_y_pred = tree_model.predict(X_test)
# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, tree_y_pred)
print(classification_report(y_true=y_test,
                            y_pred=tree_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---GAUSSIAN NAIVE BAYES---")
# initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()
# train the classifier using the training data
gnb.fit(os_data_X, os_data_y.values)
# make predictions on the testing data
gnb_y_pred = gnb.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=gnb_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---KNN---")
#try different k and check each accuracy
neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(os_data_X, os_data_y.values.ravel())
   train_accuracy[i] = knn.score(os_data_X, os_data_y.values.ravel())
   test_accuracy[i] = knn.score(X_test, y_test)

plt.figure(figsize=(10,7))
plt.title('KNN trying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# use k with highest accuracy for final prediction result
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(os_data_X,os_data_y.values.ravel())
score = knn.score(X_test,y_test)
knn_y_pred = knn.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=knn_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---LOGISTIC REGRESSION---")
logreg = LogisticRegression()
logreg.fit(os_data_X, os_data_y.values.ravel())
logreg_y_pred = logreg.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=logreg_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---LDA---")
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(os_data_X, os_data_y.values.ravel())
# Define method to evaluate model
lda_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

# evaluate model
lda_score = cross_val_score(lda_model, os_data_X, os_data_y.values.ravel(), scoring='accuracy', cv=lda_cv)
print("LDA score: ", np.mean(lda_score))
lda_y_pred = lda_model.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=lda_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---LINEAR SVM---")
# create an SVM classifier
svm_model = svm.SVC(kernel='linear')
# train the classifier on the training data
svm_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
svm_y_pred = svm_model.predict(X_test)
print(classification_report(y_true=y_test, 
                            y_pred=svm_y_pred,
                            target_names=["Not estrus", "Estrus"]))

print("---POLY SVM---")
# create an SVM classifier
svm_model = svm.SVC(kernel='poly')
# train the classifier on the training data
svm_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
svm_y_pred = svm_model.predict(X_test)
print(classification_report(y_true=y_test, 
                            y_pred=svm_y_pred,
                            target_names=["Not estrus", "Estrus"]))

print("---RBF SVM---")
# create an SVM classifier
svm_model = svm.SVC(kernel='rbf')
# train the classifier on the training data
svm_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
svm_y_pred = svm_model.predict(X_test)
print(classification_report(y_true=y_test, 
                            y_pred=svm_y_pred,
                            target_names=["Not estrus", "Estrus"]))

print("USING 13 COMPONENTS")
pca_2 = PCA(n_components=13)
X_pca_2 = pca.fit_transform(X)

print("---BALANCING DATA---")
y = gen['estrus'].values
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_pca_2, y, test_size=0.4, random_state=0)

# resample on training data
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
# balanced train data to be used
os_data_X = pd.DataFrame(data=os_data_X)
os_data_y= pd.DataFrame(data=os_data_y,columns=['estrus'])

# check the numbers of the data
print("Length of oversampled data is ",len(os_data_X))
print("Number of not estrus in oversampled data",len(os_data_y[os_data_y['estrus']==0]))
print("Number of estrus",len(os_data_y[os_data_y['estrus']==1]))
print("Proportion of not estrus data in oversampled data is ",len(os_data_y[os_data_y['estrus']==0])/len(os_data_X))
print("Proportion of estrus data in oversampled data is ",len(os_data_y[os_data_y['estrus']==1])/len(os_data_X))


print("---CLASSIFICATION TREE---")
tree_model = DecisionTreeClassifier()
# train the classifier on the training data
tree_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
tree_y_pred = tree_model.predict(X_test)
# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, tree_y_pred)
print(classification_report(y_true=y_test,
                            y_pred=tree_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---GAUSSIAN NAIVE BAYES---")
# initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()
# train the classifier using the training data
gnb.fit(os_data_X, os_data_y.values)
# make predictions on the testing data
gnb_y_pred = gnb.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=gnb_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---KNN---")
#try different k and check each accuracy
neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(os_data_X, os_data_y.values.ravel())
   train_accuracy[i] = knn.score(os_data_X, os_data_y.values.ravel())
   test_accuracy[i] = knn.score(X_test, y_test)

plt.figure(figsize=(10,7))
plt.title('KNN trying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# use k with highest accuracy for final prediction result
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(os_data_X,os_data_y.values.ravel())
score = knn.score(X_test,y_test)
knn_y_pred = knn.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=knn_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---LOGISTIC REGRESSION---")
logreg = LogisticRegression()
logreg.fit(os_data_X, os_data_y.values.ravel())
logreg_y_pred = logreg.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=logreg_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---LDA---")
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(os_data_X, os_data_y.values.ravel())
# Define method to evaluate model
lda_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

# evaluate model
lda_score = cross_val_score(lda_model, os_data_X, os_data_y.values.ravel(), scoring='accuracy', cv=lda_cv)
print("LDA score: ", np.mean(lda_score))
lda_y_pred = lda_model.predict(X_test)
print(classification_report(y_true=y_test,
                            y_pred=lda_y_pred,
                            target_names=["Not estrus", "Estrus"]))


print("---LINEAR SVM---")
# create an SVM classifier
svm_model = svm.SVC(kernel='linear')
# train the classifier on the training data
svm_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
svm_y_pred = svm_model.predict(X_test)
print(classification_report(y_true=y_test, 
                            y_pred=svm_y_pred,
                            target_names=["Not estrus", "Estrus"]))

print("---POLY SVM---")
# create an SVM classifier
svm_model = svm.SVC(kernel='poly')
# train the classifier on the training data
svm_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
svm_y_pred = svm_model.predict(X_test)
print(classification_report(y_true=y_test, 
                            y_pred=svm_y_pred,
                            target_names=["Not estrus", "Estrus"]))

print("---RBF SVM---")
# create an SVM classifier
svm_model = svm.SVC(kernel='rbf')
# train the classifier on the training data
svm_model.fit(os_data_X, os_data_y.values.ravel())
# make predictions on the test data
svm_y_pred = svm_model.predict(X_test)
print(classification_report(y_true=y_test, 
                            y_pred=svm_y_pred,
                            target_names=["Not estrus", "Estrus"]))

print("Done:)")
