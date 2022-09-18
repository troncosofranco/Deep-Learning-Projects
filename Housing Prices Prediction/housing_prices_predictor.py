#Modules
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#data loading
df = pd.read_csv("housing_prices.csv")

#Data visualization
sns.scatterplot(df['sqft_living'], df['price'])

#Correlation between the variables (heatmap)
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(), annot= True)

#data cleaning
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement' ]
X = df[selected_features]
y = df['price']

#scaling features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#Normalize output
y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)

#Train and test data set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2)

#Building model
model = tf.keras.Sequential() #building sequentially layer by layer 
model.add(tf.keras.layers.Dense(units= 100,activation = 'relu', input_shape = (7, )))  #100 neurons, 7 inputs
model.add(tf.keras.layers.Dense(units= 100,activation = 'relu'))  # hidden layer: 100 neurons
model.add(tf.keras.layers.Dense(units= 100,activation = 'relu'))  # hidden layer: 100 neurons
model.add(tf.keras.layers.Dense(units= 1,activation = 'linear')) #output layer (1 output)

#model.summary()

#Compiling model
model.compile(optimizer='Adam', loss= 'mean_squared_error')

#Training model
epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size=50, validation_split = 0.2)

#Testing model
epochs_hist.history.keys() #history keys

#Graphic
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Loss progress during training')
plt.xlabel('Epoch') 
plt.ylabel('Training loss')
plt.legend('Training loss', 'Validation loss')

#Prediction
#Inputs: ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement' ]
X_test_1 = np.array([[ 4, 3, 1960, 5000, 1, 2000, 3000 ]])

#scaling feature
scaler_1 = MinMaxScaler()
X_test_scaled_1 = scaler_1.fit_transform(X_test_1)

#Making prediction
y_predict_1 = model.predict(X_test_scaled_1)


#Inverse transform
y_predict_1 = scaler.inverse_transform(y_predict_1)
print('The price of the house for data input will be $', y_predict_1)



