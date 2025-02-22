import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("house_price_data.csv")


X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


ln = LinearRegression()
ln.fit(X_train, y_train)


size = float(input("Enter the size (sq ft): "))
bed = float(input("Enter the number of Bedrooms: "))
bath = float(input("Enter the number of Bathrooms: "))
garage = float(input("Enter the number of Garages: "))

data=[[size,bed,bath,garage]]
price_prediction=ln.predict(data)
print(price_prediction[0])
