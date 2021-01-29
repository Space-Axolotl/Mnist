import tensorflow as tf

# 28x28 pixel images of hand written digits 0-9
mnist = tf.keras.datasets.mnist 

# unpack it to train and test arrays
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize/scale matrix values into range from 0 to 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# make a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'] )

# train the model
model.fit(x_train, y_train, epochs=3)

# check accuracy on test data
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_acc, val_loss)

model.save('mnisTrained.model')