```markdown
# **Author:** Ryan Hatch  
# **Date of Development:** Mon Nov 4th, 2024  
# **Last Modified:** Tues Nov 5th, 2024  

---

**Description:** This is a simple example of a neural network using TensorFlow 2.0.

---

# To install TensorFlow 2.0 using pip:
pip install tensorflow==2.0.0
pip install --upgrade tensorflow
```

---

## Checking Tensorflow is installed correctly

```python
import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

Output:
```plaintext
TensorFlow version: 2.18.0
Keras version: 3.6.0
Num GPUs Available:  0
```

---

## Clearing the Session

```python
from tensorflow.keras import backend as K
K.clear_session()
```

**Warning:**  
Use `tf.compat.v1.reset_default_graph` instead of `tf.reset_default_graph`.

---

## Improving the Simple Net in Keras with Hidden Layers

Adding additional layers to improve accuracy.

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

np.random.seed(1671)

NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

(X_train, y_train), (X_test, y_test) = mnist.load_data()
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED).astype('float32') / 255
X_test = X_test.reshape(10000, RESHAPED).astype('float32') / 255

Y_train = to_categorical(y_train, NB_CLASSES)
Y_test = to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
```

---

## Experiment 1: Increased Epochs to 30

```python
NB_EPOCH = 30
BATCH_SIZE = 128
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

K.clear_session()
model_exp1 = Sequential()
model_exp1.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model_exp1.add(Activation('relu'))
model_exp1.add(Dense(N_HIDDEN))
model_exp1.add(Activation('relu'))
model_exp1.add(Dense(NB_CLASSES))
model_exp1.add(Activation('softmax'))

optimizer_exp1 = SGD()
model_exp1.compile(loss='categorical_crossentropy', optimizer=optimizer_exp1, metrics=['accuracy'])

history_exp1 = model_exp1.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score_exp1 = model_exp1.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Experiment 1 - Test score:", score_exp1[0])
print("Experiment 1 - Test accuracy:", score_exp1[1])
```

---

## Experiment 2: Decreased Epochs to 10

```python
NB_EPOCH = 10
BATCH_SIZE = 128
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

K.clear_session()
model_exp2 = Sequential()
model_exp2.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model_exp2.add(Activation('relu'))
model_exp2.add(Dense(N_HIDDEN))
model_exp2.add(Activation('relu'))
model_exp2.add(Dense(NB_CLASSES))
model_exp2.add(Activation('softmax'))

optimizer_exp2 = SGD()
model_exp2.compile(loss='categorical_crossentropy', optimizer=optimizer_exp2, metrics=['accuracy'])

history_exp2 = model_exp2.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score_exp2 = model_exp2.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Experiment 2 - Test score:", score_exp2[0])
print("Experiment 2 - Test accuracy:", score_exp2[1])
```

---

## Experiment 3: Increased Hidden Neurons to 256

```python
NB_EPOCH = 20
BATCH_SIZE = 128
N_HIDDEN = 256
VALIDATION_SPLIT = 0.2

K.clear_session()
model_exp3 = Sequential()
model_exp3.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model_exp3.add(Activation('relu'))
model_exp3.add(Dense(N_HIDDEN))
model_exp3.add(Activation('relu'))
model_exp3.add(Dense(NB_CLASSES))
model_exp3.add(Activation('softmax'))

model_exp3.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

history_exp3 = model_exp3.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score_exp3 = model_exp3.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Experiment 3 - Test score:", score_exp3[0])
print("Experiment 3 - Test accuracy:", score_exp3[1])
```

---
