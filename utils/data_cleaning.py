import pandas as pd
import pickle 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def create_model(data):
    le = LabelEncoder()
    scaler = StandardScaler()
    data['species'] = le.fit_transform(data['species'])
    x = data.drop('species', axis=1)
    y = data['species']
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)    

    svm = SVC(C = 10, gamma = 'scale', kernel = 'linear', probability=True)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    print("SVM Classification Report:\n", classification_report(y_test, y_pred))
    
    return svm

def main():
    data = pd.read_csv('E:\\23881A66E2\\Projects\\Iris_Flower_Classification\\data\\IRIS.csv')    
    model = create_model(data)

    with open("E:\\23881A66E2\\Projects\\Iris_Flower_Classification\\model\\svm_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()

