import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def fish_weight_predictor():
    training_dataFrame = pd.read_csv('Fish.csv')
    
    training_dataFrame['Species']=training_dataFrame['Species'].replace({'Bream': 0, 'Parkki': 1, 'Perch': 2, 'Pike': 3, 'Roach': 4, 'Smelt': 5, 'Whitefish': 6}) # Converting Categorical Data to numerical data

    X = training_dataFrame.drop('Weight',axis=1) # Features
    y = training_dataFrame['Weight']  # Label

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f'Predictions: {y_pred}')

    joblib.dump(model, 'fish_weight_predictor.pkl')


if __name__ =='__main__':
    fish_weight_predictor()