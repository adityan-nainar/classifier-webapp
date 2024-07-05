import streamlit as st
import numpy as np

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# st.write('##### Streamlit application')

st.write(""" # Explore different classifiers """)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer": 
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    
    X = data.data
    y = data.target

    return X,y

X,y = get_dataset(dataset_name)

def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == "KNN":
        k = st.sidebar.slider("K",1,15)
        params["K"] = k
    elif clf_name == "SVM":
        c = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = c
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf  =SVC(C=params["C"]) 
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], random_state=1000)

    return clf

clf = get_classifier(classifier_name, params)

#Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.write("#### Which one is the best?")
    st.write("Shape of dataset", X.shape)
    st.write("Number of classes", len(np.unique(y)))

with col2:
    st.markdown(f"### **Classifier: {classifier_name}**")
    st.write(f"Accuracy: {accuracy}")

#Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2,c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig=fig)