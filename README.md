# Training and Comparing Neural Network Models

#### **Author:** Ryan Hatch  
> **Date of Development:** Mon Nov 4th, 2024  
> **Last Modified:** Tues Nov 5th, 2024  
> **Description:** Building a neural network using TensorFlow 2.0 and comparing the accuracy of the outputs based off of the models training.

---

#### To install TensorFlow 2.0 using pip:
```markdown
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
```
60000 train samples
10000 test samples

Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 128)            │       100,480 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation (Activation)         │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 128)            │        16,512 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_1 (Activation)       │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 10)             │         1,290 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_2 (Activation)       │ (None, 10)             │             0 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 118,282 (462.04 KB)

 Trainable params: 118,282 (462.04 KB)

 Non-trainable params: 0 (0.00 B)

Epoch 1/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.4416 - loss: 1.9155 - val_accuracy: 0.8271 - val_loss: 0.7860
Epoch 2/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8250 - loss: 0.7064 - val_accuracy: 0.8775 - val_loss: 0.4707
Epoch 3/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8677 - loss: 0.4762 - val_accuracy: 0.8944 - val_loss: 0.3829
Epoch 4/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8854 - loss: 0.4056 - val_accuracy: 0.9039 - val_loss: 0.3426
Epoch 5/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8985 - loss: 0.3614 - val_accuracy: 0.9096 - val_loss: 0.3168
Epoch 6/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9067 - loss: 0.3279 - val_accuracy: 0.9158 - val_loss: 0.2979
Epoch 7/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9108 - loss: 0.3154 - val_accuracy: 0.9204 - val_loss: 0.2830
Epoch 8/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9154 - loss: 0.3001 - val_accuracy: 0.9217 - val_loss: 0.2716
Epoch 9/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9211 - loss: 0.2822 - val_accuracy: 0.9271 - val_loss: 0.2601
Epoch 10/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9233 - loss: 0.2759 - val_accuracy: 0.9289 - val_loss: 0.2501
Epoch 11/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9265 - loss: 0.2607 - val_accuracy: 0.9306 - val_loss: 0.2428
Epoch 12/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9307 - loss: 0.2451 - val_accuracy: 0.9333 - val_loss: 0.2359
Epoch 13/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9320 - loss: 0.2386 - val_accuracy: 0.9352 - val_loss: 0.2280
Epoch 14/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9360 - loss: 0.2307 - val_accuracy: 0.9366 - val_loss: 0.2230
Epoch 15/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9361 - loss: 0.2260 - val_accuracy: 0.9400 - val_loss: 0.2143
Epoch 16/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9395 - loss: 0.2138 - val_accuracy: 0.9421 - val_loss: 0.2095
Epoch 17/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9388 - loss: 0.2153 - val_accuracy: 0.9437 - val_loss: 0.2031
Epoch 18/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9424 - loss: 0.2021 - val_accuracy: 0.9448 - val_loss: 0.1985
Epoch 19/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9452 - loss: 0.1937 - val_accuracy: 0.9474 - val_loss: 0.1951
Epoch 20/20
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9439 - loss: 0.1940 - val_accuracy: 0.9482 - val_loss: 0.1894
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9368 - loss: 0.2182
Test score: 0.18993066251277924
Test accuracy: 0.9442999958992004
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
```
Epoch 1/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.4659 - loss: 1.8417 - val_accuracy: 0.8301 - val_loss: 0.7231
Epoch 2/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.8315 - loss: 0.6646 - val_accuracy: 0.8858 - val_loss: 0.4531
Epoch 3/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.8782 - loss: 0.4614 - val_accuracy: 0.8992 - val_loss: 0.3711
Epoch 4/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.8940 - loss: 0.3883 - val_accuracy: 0.9052 - val_loss: 0.3319
Epoch 5/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9025 - loss: 0.3476 - val_accuracy: 0.9134 - val_loss: 0.3071
Epoch 6/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9107 - loss: 0.3190 - val_accuracy: 0.9175 - val_loss: 0.2908
Epoch 7/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9149 - loss: 0.2980 - val_accuracy: 0.9208 - val_loss: 0.2771
Epoch 8/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9160 - loss: 0.2930 - val_accuracy: 0.9236 - val_loss: 0.2662
Epoch 9/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9204 - loss: 0.2760 - val_accuracy: 0.9268 - val_loss: 0.2559
Epoch 10/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9256 - loss: 0.2636 - val_accuracy: 0.9284 - val_loss: 0.2479
Epoch 11/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9280 - loss: 0.2518 - val_accuracy: 0.9316 - val_loss: 0.2387
Epoch 12/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9280 - loss: 0.2505 - val_accuracy: 0.9342 - val_loss: 0.2322
Epoch 13/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9309 - loss: 0.2434 - val_accuracy: 0.9355 - val_loss: 0.2252
Epoch 14/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9332 - loss: 0.2348 - val_accuracy: 0.9382 - val_loss: 0.2184
Epoch 15/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9375 - loss: 0.2205 - val_accuracy: 0.9392 - val_loss: 0.2146
Epoch 16/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9399 - loss: 0.2144 - val_accuracy: 0.9422 - val_loss: 0.2081
Epoch 17/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9384 - loss: 0.2154 - val_accuracy: 0.9440 - val_loss: 0.2024
Epoch 18/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9422 - loss: 0.2020 - val_accuracy: 0.9457 - val_loss: 0.1983
Epoch 19/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9429 - loss: 0.1951 - val_accuracy: 0.9448 - val_loss: 0.1962
Epoch 20/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9454 - loss: 0.1904 - val_accuracy: 0.9467 - val_loss: 0.1896
Epoch 21/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9471 - loss: 0.1853 - val_accuracy: 0.9469 - val_loss: 0.1857
Epoch 22/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9461 - loss: 0.1850 - val_accuracy: 0.9498 - val_loss: 0.1819
Epoch 23/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9494 - loss: 0.1762 - val_accuracy: 0.9507 - val_loss: 0.1780
Epoch 24/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9496 - loss: 0.1755 - val_accuracy: 0.9489 - val_loss: 0.1753
Epoch 25/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9526 - loss: 0.1675 - val_accuracy: 0.9509 - val_loss: 0.1714
Epoch 26/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.9539 - loss: 0.1635 - val_accuracy: 0.9521 - val_loss: 0.1682
Epoch 27/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9544 - loss: 0.1632 - val_accuracy: 0.9541 - val_loss: 0.1662
Epoch 28/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9548 - loss: 0.1571 - val_accuracy: 0.9539 - val_loss: 0.1634
Epoch 29/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9562 - loss: 0.1527 - val_accuracy: 0.9550 - val_loss: 0.1596
Epoch 30/30
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9577 - loss: 0.1509 - val_accuracy: 0.9563 - val_loss: 0.1572
[1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.9470 - loss: 0.1808
Experiment 1 - Test score: 0.15435431897640228
Experiment 1 - Test accuracy: 0.954800009727478
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
```
Epoch 1/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.4725 - loss: 1.8071 - val_accuracy: 0.8499 - val_loss: 0.7077
Epoch 2/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.8429 - loss: 0.6510 - val_accuracy: 0.8873 - val_loss: 0.4469
Epoch 3/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.8781 - loss: 0.4551 - val_accuracy: 0.8996 - val_loss: 0.3709
Epoch 4/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.8913 - loss: 0.3883 - val_accuracy: 0.9084 - val_loss: 0.3345
Epoch 5/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9007 - loss: 0.3509 - val_accuracy: 0.9123 - val_loss: 0.3117
Epoch 6/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9053 - loss: 0.3360 - val_accuracy: 0.9162 - val_loss: 0.2943
Epoch 7/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9109 - loss: 0.3107 - val_accuracy: 0.9195 - val_loss: 0.2811
Epoch 8/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9162 - loss: 0.2934 - val_accuracy: 0.9224 - val_loss: 0.2699
Epoch 9/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9199 - loss: 0.2836 - val_accuracy: 0.9252 - val_loss: 0.2593
Epoch 10/10
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9227 - loss: 0.2709 - val_accuracy: 0.9279 - val_loss: 0.2506
[1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.9157 - loss: 0.2899
Experiment 2 - Test score: 0.24992172420024872
Experiment 2 - Test accuracy: 0.9279000163078308
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
```
Epoch 1/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.4952 - loss: 1.8586 - val_accuracy: 0.8568 - val_loss: 0.6780
Epoch 2/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.8578 - loss: 0.6135 - val_accuracy: 0.8911 - val_loss: 0.4237
Epoch 3/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.8863 - loss: 0.4281 - val_accuracy: 0.9025 - val_loss: 0.3557
Epoch 4/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.8989 - loss: 0.3680 - val_accuracy: 0.9113 - val_loss: 0.3196
Epoch 5/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9084 - loss: 0.3277 - val_accuracy: 0.9172 - val_loss: 0.2969
Epoch 6/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9140 - loss: 0.3034 - val_accuracy: 0.9211 - val_loss: 0.2798
Epoch 7/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9182 - loss: 0.2869 - val_accuracy: 0.9258 - val_loss: 0.2654
Epoch 8/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9229 - loss: 0.2725 - val_accuracy: 0.9280 - val_loss: 0.2548
Epoch 9/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9252 - loss: 0.2613 - val_accuracy: 0.9317 - val_loss: 0.2457
Epoch 10/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9284 - loss: 0.2498 - val_accuracy: 0.9333 - val_loss: 0.2349
Epoch 11/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9305 - loss: 0.2408 - val_accuracy: 0.9356 - val_loss: 0.2278
Epoch 12/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9316 - loss: 0.2359 - val_accuracy: 0.9381 - val_loss: 0.2215
Epoch 13/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9375 - loss: 0.2184 - val_accuracy: 0.9409 - val_loss: 0.2138
Epoch 14/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9373 - loss: 0.2163 - val_accuracy: 0.9426 - val_loss: 0.2073
Epoch 15/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9408 - loss: 0.2033 - val_accuracy: 0.9455 - val_loss: 0.2003
Epoch 16/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9415 - loss: 0.1984 - val_accuracy: 0.9467 - val_loss: 0.1955
Epoch 17/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9436 - loss: 0.1969 - val_accuracy: 0.9478 - val_loss: 0.1910
Epoch 18/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9484 - loss: 0.1876 - val_accuracy: 0.9498 - val_loss: 0.1857
Epoch 19/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9482 - loss: 0.1853 - val_accuracy: 0.9503 - val_loss: 0.1820
Epoch 20/20
[1m375/375[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9466 - loss: 0.1813 - val_accuracy: 0.9519 - val_loss: 0.1782
[1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.9393 - loss: 0.2069
Experiment 3 - Test score: 0.17764912545681
Experiment 3 - Test accuracy: 0.9480000138282776
```
---
