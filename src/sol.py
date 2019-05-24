#%%
import cv2
import csv
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
#%%
TRAIN_FOLDER = "/home/tanaka/Downloads/bonus_sml/sml_train/train_"
TRAIN_LABEL = "/home/tanaka/Downloads/bonus_sml/sml_train.csv"
TEST_FOLDER = "/home/tanaka/Downloads/bonus_sml/sml_validation/val_"
TEST_LABEL = "/home/tanaka/Downloads/bonus_sml/sml_val.csv"
#%%
X_train = []
y_train = []
X_test = []
y_test = []
#%%
print("IMPORTING TRAIN")
#%%
for i in range(10000):
    f = TRAIN_FOLDER + str(i) + ".jpg"
    im = cv2.imread(f).flatten()
    X_train.append(im)
#%%
print("IMPORTING TEST")
#%%
for i in range(1000):
    f = TEST_FOLDER + str(i) + ".jpg"
    im = cv2.imread(f).flatten()
    X_test.append(im)
#%%
print("IMPORTING TRAIN LABEL")
#%%
with open(TRAIN_LABEL, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    head = csvreader.__next__()
    for row in csvreader:
        y_train.append(row[1])
csvfile.close()
#%%
print("IMPORTING TEST LABEL")
#%%
with open(TEST_LABEL, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        print(len(row))
        print(row[0])
        print(row[1])
        y_test.append(row[1])
csvfile.close()
#%%
print("RUNNING PCA")
#%%
pca = PCA(n_components=15)
pca.fit(X_train)
#%%
print("TRANSFORMING")
#%%
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
#%%
print("RUNNING RF")
#%%
clf = RandomForestClassifier(n_estimators=500, verbose=2)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
#%%
import pickle
with open('sth.pkl', 'wb+') as f:
    pickle.dump(y_predict, f)
#%%
print(len(y_test))
print(len(y_predict))
from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test, y_predict))