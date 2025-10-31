# iris_classification.py
# Decision Tree classifier on Iris dataset with a small plot saved to file.
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

def main():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    clf = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,y_pred))
    print("\nClassification Report:\n", classification_report(y_test,y_pred))
    # save a simple scatter of two features
    plt.figure(figsize=(6,4))
    for cls in np.unique(y):
        idx = y==cls
        plt.scatter(X[idx,0], X[idx,1], label=data.target_names[cls])
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.legend()
    plt.title('Iris dataset (feature 0 vs 1)')
    plt.savefig('iris_scatter.png')
    print('Saved iris_scatter.png')

if __name__ == '__main__':
    main()