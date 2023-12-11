import pandas as pd
import tensorflow as tf

data = pd.read_csv("lemonade.csv")

x1 = data[["온도"]]
y = data[["판매량"]]

X = tf.keras.layers.Input(shape = [1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss = "mse")

model.fit(x1, y, epochs = 2000)

print("Prediction :", model.predict([[15]]))