import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#data loading
df = pd.read_csv("ice_cream_sales.csv")

#data visualization
sns.scatterplot(df['Temperature'], df['Revenue'])

#Train dataset
X_train = df['Temperature']
y_train = df['Revenue']

#Building model
model = tf.keras.Sequential() #building sequentially layer by layer 
model.add(tf.keras.layers.Dense(units= 1, input_shape = [1])) #adding input layer dense. one neuron.

#model.summary() #information of NN layer

#Compiling model
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss= 'mean_squared_error')

#Training model
epochs_hist = model.fit(X_train, y_train, epochs = 20 )

#Testing model
keys = epochs_hist.history.keys() #history keys


#graphics
plt.plot(epochs_hist.history['loss'])
plt.title('Loss progress during training')
plt.xlabel('Epoch') 
plt.ylabel('Training loss')
plt.legend(['Training loss'])

#Getting model weights
weights = model.get_weights()
print(weights) #parameters or linear regression

#Predictions
temperature = 5
revenue = model.predict([temperature])
print('The revenue will be:',revenue)


#prediction graphic
plt.scatter(X_train, y_train, color='grey')
plt.plot(X_train, model.predict(X_train), color='red')
plt.xlabel('Temperature (°C)') 
plt.ylabel('Revenue (U$d)')
plt.title('Revenue vs Ambient temperature')
plt.show()