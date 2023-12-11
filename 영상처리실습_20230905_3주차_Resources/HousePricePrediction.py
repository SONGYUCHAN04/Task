import tensorflow as tf
import pandas as pd

data = pd.read_csv("boston.csv")

x = data[["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]]
y = data[["medv"]]

X = tf.keras.layers.Input(shape = [13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss = "mse")

# model.fit(x, y, epochs = 1000)

# print(model.get_weights())

print(model.predict(x[5 : 10]))
print(y[5 : 10])