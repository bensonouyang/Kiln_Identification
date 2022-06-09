```python
import tensorflow as tf
import imageio
import os
from PIL import Image
import pandas as pd
import numpy as np
```


```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images, test_labels) = fashion_mnist.load_data()
```


```python
N = tr.shape[0]
```


```python
tr = pd.read_csv('Ytr.txt')
te = pd.read_csv('pred.txt')
```


```python
te
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>618</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>620</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>623</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>625</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>629</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2509</th>
      <td>6141</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2510</th>
      <td>6143</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2511</th>
      <td>6144</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2512</th>
      <td>6146</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2513</th>
      <td>6147</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2514 rows × 2 columns</p>
</div>




```python
D = 28 # Dimension of images after resizing
Xtr = np.zeros([N,D,D])
```


```python
ii = 0
path = 'images/%05d.png' %ii
        
pic = imageio.imread(path)

pic.shape
```




    (256, 256, 3)




```python
found = list()
for ii in range(N):
    if ii % 100 == 0:
        print('%d / %d' % (ii, N))

    path = 'images/%05d.png' %ii
        
    pic = imageio.imread(path)
    pic = Image.fromarray(pic).resize((D,D)) # NOTE : this can be improved.
    pic = np.mean(pic,axis = 2) # NOTE: this can be improved
    pic = np.array(pic)
    # print(pic.shape)
    Xtr[ii,:,:] = pic
```

    0 / 3634
    100 / 3634
    200 / 3634
    300 / 3634
    400 / 3634
    500 / 3634
    600 / 3634
    700 / 3634
    800 / 3634
    900 / 3634
    1000 / 3634
    1100 / 3634
    1200 / 3634
    1300 / 3634
    1400 / 3634
    1500 / 3634
    1600 / 3634
    1700 / 3634
    1800 / 3634
    1900 / 3634
    2000 / 3634
    2100 / 3634
    2200 / 3634
    2300 / 3634
    2400 / 3634
    2500 / 3634
    2600 / 3634
    2700 / 3634
    2800 / 3634
    2900 / 3634
    3000 / 3634
    3100 / 3634
    3200 / 3634
    3300 / 3634
    3400 / 3634
    3500 / 3634
    3600 / 3634
    


```python
Xtr2 = Xtr/255
```


```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (D,D)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(2) # output neuron for number of classes
])
```


```python
model.compile(optimizer = 'adam', 
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
```


```python
model.fit(Xtr2, np.array(tr['label']), epochs = 10) # guessing 0 have higher chance than accuracy

# DNN with 28x28 & N = 1000 yields 0.64
# DNN with 28x28 & N = tr.shape[0] yields 0.53
# Continue?
# DNN with 28x28 & N = tr.shape[0] , epochs = 1000 yields
# DNN with 50x50 & N = tr.shape[0] epochs = 1000 yields 

# or
# Look at something simpler - how much orange in image?
# or
# use cnn (convolutional) https://blog.tensorflow.org/2018/04/fashion-mnist-with-tfkeras.html
```

    Epoch 1/10
    114/114 [==============================] - 0s 4ms/step - loss: 0.6633 - accuracy: 0.6189
    Epoch 2/10
    114/114 [==============================] - 0s 3ms/step - loss: 0.6564 - accuracy: 0.6189
    Epoch 3/10
    114/114 [==============================] - 0s 4ms/step - loss: 0.6545 - accuracy: 0.6197
    Epoch 4/10
    114/114 [==============================] - 0s 3ms/step - loss: 0.6566 - accuracy: 0.6186
    Epoch 5/10
    114/114 [==============================] - 0s 3ms/step - loss: 0.6562 - accuracy: 0.6150
    Epoch 6/10
    114/114 [==============================] - 0s 2ms/step - loss: 0.6526 - accuracy: 0.6216
    Epoch 7/10
    114/114 [==============================] - 0s 3ms/step - loss: 0.6516 - accuracy: 0.6255
    Epoch 8/10
    114/114 [==============================] - 0s 2ms/step - loss: 0.6537 - accuracy: 0.6170
    Epoch 9/10
    114/114 [==============================] - 0s 2ms/step - loss: 0.6506 - accuracy: 0.6214
    Epoch 10/10
    114/114 [==============================] - 0s 3ms/step - loss: 0.6548 - accuracy: 0.6236
    




    <tensorflow.python.keras.callbacks.History at 0x272e6089e20>




```python
te['id']
```




    0        618
    1        620
    2        623
    3        625
    4        629
            ... 
    2509    6141
    2510    6143
    2511    6144
    2512    6146
    2513    6147
    Name: id, Length: 2514, dtype: int64




```python
D = 28 # Dimension of images after resizing
N = te.shape[0]
Xte = np.zeros([N,D,D])
found = list()
for ii in range(N):
    if ii % 100 == 0:
        print('%d / %d' % (ii, N))
    teId = te['id'][ii]
    path = 'images/%05d.png' %teId
        
    pic = imageio.imread(path)
    pic = Image.fromarray(pic).resize((D,D)) # NOTE : this can be improved.
    pic = np.mean(pic,axis = 2) # NOTE: this can be improved
    pic = np.array(pic)
    # print(pic.shape)
    Xte[ii,:,:] = pic
```

    0 / 2514
    100 / 2514
    200 / 2514
    300 / 2514
    400 / 2514
    500 / 2514
    600 / 2514
    700 / 2514
    800 / 2514
    900 / 2514
    1000 / 2514
    1100 / 2514
    1200 / 2514
    1300 / 2514
    1400 / 2514
    1500 / 2514
    1600 / 2514
    1700 / 2514
    1800 / 2514
    1900 / 2514
    2000 / 2514
    2100 / 2514
    2200 / 2514
    2300 / 2514
    2400 / 2514
    2500 / 2514
    


```python
Xte2 = Xte/255
```


```python
preds = model.predict(Xte2)
```


```python
preds
```




    array([[-0.13217375, -0.23582153],
           [ 0.31179634, -0.33921182],
           [ 0.25503376, -0.3952964 ],
           ...,
           [ 0.07474652, -0.3258608 ],
           [ 0.02470782, -0.2948069 ],
           [ 0.39199814, -0.40421438]], dtype=float32)




```python
pred = tf.nn.sigmoid(preds)
pred = tf.where(pred<0.5, 0 , 1)
```


```python

```




    array([[1, 1],
           [1, 0],
           [1, 0],
           ...,
           [1, 0],
           [1, 1],
           [1, 0]])




```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(Xtr2,tr['label'],test_size = 0.2, random_state = 42)
```


```python
model.fit(X_train, y_train, epochs = 5, validation_data = (X_test,y_test))
```

    Epoch 1/5
    91/91 [==============================] - 0s 5ms/step - loss: 0.6500 - accuracy: 0.6202 - val_loss: 0.6632 - val_accuracy: 0.5915
    Epoch 2/5
    91/91 [==============================] - 0s 4ms/step - loss: 0.6537 - accuracy: 0.6175 - val_loss: 0.6538 - val_accuracy: 0.6190
    Epoch 3/5
    91/91 [==============================] - 0s 4ms/step - loss: 0.6530 - accuracy: 0.6151 - val_loss: 0.6688 - val_accuracy: 0.6245
    Epoch 4/5
    91/91 [==============================] - 0s 4ms/step - loss: 0.6513 - accuracy: 0.6271 - val_loss: 0.6632 - val_accuracy: 0.5846
    Epoch 5/5
    91/91 [==============================] - 0s 3ms/step - loss: 0.6493 - accuracy: 0.6237 - val_loss: 0.6641 - val_accuracy: 0.6011
    




    <tensorflow.python.keras.callbacks.History at 0x272e6436310>




```python

```




    1289    0
    120     0
    839     1
    1537    0
    2890    0
           ..
    343     0
    1652    0
    1592    0
    678     0
    1831    0
    Name: label, Length: 727, dtype: int64




```python
model.predict(X_test)
```




    array([[-0.0930413 , -0.45244193],
           [ 0.0761469 , -0.3376736 ],
           [ 0.01296997, -0.45653886],
           ...,
           [-0.00528729, -0.4273917 ],
           [ 0.11757076, -0.40412998],
           [-0.41486728, -0.36487132]], dtype=float32)




```python
def read_image(N,D,X):
    Xt = np.zeros([N,D,D])
    #found = list()
    for ii in range(N):
        if ii % 100 == 0:
            print('%d / %d' % (ii, N))
        teId = X['id'][ii]
        path = 'images/%05d.png' %teId
        
        pic = imageio.imread(path)
        pic = Image.fromarray(pic).resize((D,D)) # NOTE : this can be improved.
        pic = np.mean(pic,axis = 2) # NOTE: this can be improved
        pic = np.array(pic)
        # print(pic.shape)
        Xt[ii,:,:] = pic
    Xt = Xt/255
    return Xt
```


```python
Xtr_1 = read_image(tr.shape[0],200,tr)
```

    0 / 3634
    100 / 3634
    200 / 3634
    300 / 3634
    400 / 3634
    500 / 3634
    600 / 3634
    700 / 3634
    800 / 3634
    900 / 3634
    1000 / 3634
    1100 / 3634
    1200 / 3634
    1300 / 3634
    1400 / 3634
    1500 / 3634
    1600 / 3634
    1700 / 3634
    1800 / 3634
    1900 / 3634
    2000 / 3634
    2100 / 3634
    2200 / 3634
    2300 / 3634
    2400 / 3634
    2500 / 3634
    2600 / 3634
    2700 / 3634
    2800 / 3634
    2900 / 3634
    3000 / 3634
    3100 / 3634
    3200 / 3634
    3300 / 3634
    3400 / 3634
    3500 / 3634
    3600 / 3634
    


```python
Xtr_1
```




    array([[[0.81568627, 0.8       , 0.78823529, ..., 0.25490196,
             0.26666667, 0.27843137],
            [0.85098039, 0.84705882, 0.81568627, ..., 0.23529412,
             0.22352941, 0.27843137],
            [0.77254902, 0.81960784, 0.80392157, ..., 0.27843137,
             0.25490196, 0.28627451],
            ...,
            [0.28627451, 0.2745098 , 0.26666667, ..., 0.25490196,
             0.27843137, 0.25882353],
            [0.27058824, 0.25098039, 0.26666667, ..., 0.27843137,
             0.30588235, 0.25882353],
            [0.28235294, 0.2745098 , 0.24705882, ..., 0.24705882,
             0.31372549, 0.24313725]],
    
           [[0.41176471, 0.33594771, 0.32026144, ..., 0.45359477,
             0.46013072, 0.46797386],
            [0.45751634, 0.32418301, 0.25751634, ..., 0.44052288,
             0.44836601, 0.45620915],
            [0.48104575, 0.42222222, 0.37385621, ..., 0.44444444,
             0.45228758, 0.46013072],
            ...,
            [0.37647059, 0.36470588, 0.36078431, ..., 0.30326797,
             0.39738562, 0.25228758],
            [0.39215686, 0.29411765, 0.27843137, ..., 0.32287582,
             0.32287582, 0.33856209],
            [0.40784314, 0.3254902 , 0.22745098, ..., 0.33856209,
             0.31503268, 0.33071895]],
    
           [[0.67843137, 0.67843137, 0.68627451, ..., 0.58823529,
             0.58431373, 0.58431373],
            [0.67843137, 0.68235294, 0.69019608, ..., 0.59477124,
             0.5869281 , 0.58431373],
            [0.68235294, 0.68627451, 0.69019608, ..., 0.60261438,
             0.59477124, 0.58300654],
            ...,
            [0.60915033, 0.63660131, 0.64052288, ..., 0.66143791,
             0.64183007, 0.62222222],
            [0.59738562, 0.62875817, 0.63660131, ..., 0.6496732 ,
             0.64183007, 0.62614379],
            [0.61699346, 0.63660131, 0.63660131, ..., 0.64183007,
             0.6379085 , 0.63398693]],
    
           ...,
    
           [[0.41960784, 0.40784314, 0.41176471, ..., 0.39215686,
             0.36470588, 0.36078431],
            [0.4       , 0.41960784, 0.43529412, ..., 0.38823529,
             0.38039216, 0.35686275],
            [0.36862745, 0.39607843, 0.41176471, ..., 0.37647059,
             0.39215686, 0.36078431],
            ...,
            [0.41437908, 0.40261438, 0.41437908, ..., 0.38954248,
             0.37777778, 0.38300654],
            [0.41045752, 0.39869281, 0.41045752, ..., 0.37385621,
             0.36601307, 0.39607843],
            [0.39869281, 0.39477124, 0.39869281, ..., 0.33464052,
             0.33333333, 0.40915033]],
    
           [[0.46797386, 0.49542484, 0.49934641, ..., 0.31633987,
             0.32679739, 0.29934641],
            [0.48366013, 0.50718954, 0.50588235, ..., 0.32679739,
             0.31111111, 0.28235294],
            [0.49934641, 0.51895425, 0.51633987, ..., 0.32679739,
             0.29150327, 0.27189542],
            ...,
            [0.5124183 , 0.50457516, 0.49673203, ..., 0.2496732 ,
             0.2627451 , 0.26928105],
            [0.51633987, 0.50849673, 0.50457516, ..., 0.26928105,
             0.26666667, 0.27973856],
            [0.49673203, 0.50457516, 0.5124183 , ..., 0.27320261,
             0.26666667, 0.28366013]],
    
           [[0.3372549 , 0.31764706, 0.30980392, ..., 0.45359477,
             0.44575163, 0.43398693],
            [0.3372549 , 0.31764706, 0.30980392, ..., 0.45490196,
             0.44183007, 0.43267974],
            [0.32941176, 0.31372549, 0.30980392, ..., 0.45882353,
             0.45359477, 0.4496732 ],
            ...,
            [0.3254902 , 0.32156863, 0.32156863, ..., 0.33202614,
             0.33594771, 0.32810458],
            [0.32156863, 0.32156863, 0.3254902 , ..., 0.33594771,
             0.33594771, 0.32418301],
            [0.32156863, 0.32156863, 0.3254902 , ..., 0.32810458,
             0.32810458, 0.32026144]]])




```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape = (200,200,1)),
    tf.keras.layers.MaxPool2D(2,2),
    
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid') # output neuron for number of classes
])
```


```python
model.compile(loss = 'binary_crossentropy',
             optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001),
              metrics = ['accuracy'])
```


```python
Xtr_2 = Xtr_1.reshape(-1,200,200,1)
```


```python
model.fit(Xtr_2,np.array(tr['label']), epochs = 30)
```

    Epoch 1/30
    114/114 [==============================] - 11s 53ms/step - loss: 0.8331 - accuracy: 0.6021
    Epoch 2/30
    114/114 [==============================] - 6s 49ms/step - loss: 0.6842 - accuracy: 0.6010
    Epoch 3/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.6656 - accuracy: 0.6156
    Epoch 4/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.6500 - accuracy: 0.6249
    Epoch 5/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.6343 - accuracy: 0.6511 0s - loss: 0.638
    Epoch 6/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.6023 - accuracy: 0.6835
    Epoch 7/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.5653 - accuracy: 0.7133
    Epoch 8/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.5135 - accuracy: 0.7427
    Epoch 9/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.4590 - accuracy: 0.7870
    Epoch 10/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.3866 - accuracy: 0.8217
    Epoch 11/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.3418 - accuracy: 0.8514
    Epoch 12/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.2799 - accuracy: 0.8825
    Epoch 13/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.2225 - accuracy: 0.9078
    Epoch 14/30
    114/114 [==============================] - 6s 53ms/step - loss: 0.1870 - accuracy: 0.9293
    Epoch 15/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.1479 - accuracy: 0.9499
    Epoch 16/30
    114/114 [==============================] - 6s 51ms/step - loss: 0.1582 - accuracy: 0.9587
    Epoch 17/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.1384 - accuracy: 0.9678
    Epoch 18/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.0837 - accuracy: 0.9744
    Epoch 19/30
    114/114 [==============================] - 7s 58ms/step - loss: 0.1320 - accuracy: 0.9717
    Epoch 20/30
    114/114 [==============================] - 7s 59ms/step - loss: 0.1135 - accuracy: 0.9780
    Epoch 21/30
    114/114 [==============================] - 6s 55ms/step - loss: 0.1079 - accuracy: 0.9736
    Epoch 22/30
    114/114 [==============================] - 7s 58ms/step - loss: 0.0581 - accuracy: 0.9843
    Epoch 23/30
    114/114 [==============================] - 7s 58ms/step - loss: 0.0542 - accuracy: 0.9854
    Epoch 24/30
    114/114 [==============================] - 6s 57ms/step - loss: 0.0729 - accuracy: 0.9799
    Epoch 25/30
    114/114 [==============================] - 6s 54ms/step - loss: 0.0457 - accuracy: 0.9865
    Epoch 26/30
    114/114 [==============================] - 6s 55ms/step - loss: 0.0721 - accuracy: 0.9794
    Epoch 27/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.0768 - accuracy: 0.9827
    Epoch 28/30
    114/114 [==============================] - 6s 57ms/step - loss: 0.0569 - accuracy: 0.9829
    Epoch 29/30
    114/114 [==============================] - 6s 52ms/step - loss: 0.0691 - accuracy: 0.9821
    Epoch 30/30
    114/114 [==============================] - 6s 54ms/step - loss: 0.0626 - accuracy: 0.9862
    




    <tensorflow.python.keras.callbacks.History at 0x272f3f67c10>




```python
Xtr_2.shape
```




    (3634, 200, 200, 1)




```python
X_train, X_test, y_train, y_test = train_test_split(Xtr_2,np.array(tr['label']),test_size = 0.2, random_state = 42)
```


```python
model.fit(X_train,y_train, epochs = 40, validation_data = (X_test,y_test))
```

    Epoch 1/40
    91/91 [==============================] - 6s 64ms/step - loss: 0.0436 - accuracy: 0.9859 - val_loss: 0.0266 - val_accuracy: 0.9945
    Epoch 2/40
    91/91 [==============================] - 5s 54ms/step - loss: 0.0512 - accuracy: 0.9835 - val_loss: 0.0179 - val_accuracy: 0.9959
    Epoch 3/40
    91/91 [==============================] - 5s 54ms/step - loss: 0.0440 - accuracy: 0.9880 - val_loss: 0.0206 - val_accuracy: 0.9959
    Epoch 4/40
    91/91 [==============================] - 5s 54ms/step - loss: 0.0546 - accuracy: 0.9835 - val_loss: 0.0624 - val_accuracy: 0.9945
    Epoch 5/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0345 - accuracy: 0.9900 - val_loss: 0.0700 - val_accuracy: 0.9780
    Epoch 6/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0748 - accuracy: 0.9804 - val_loss: 0.0349 - val_accuracy: 0.9931
    Epoch 7/40
    91/91 [==============================] - 6s 62ms/step - loss: 0.0496 - accuracy: 0.9849 - val_loss: 0.0395 - val_accuracy: 0.9904
    Epoch 8/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0471 - accuracy: 0.9866 - val_loss: 0.1300 - val_accuracy: 0.9464
    Epoch 9/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.0601 - accuracy: 0.9821 - val_loss: 0.0901 - val_accuracy: 0.9807
    Epoch 10/40
    91/91 [==============================] - 5s 57ms/step - loss: 0.0515 - accuracy: 0.9886 - val_loss: 0.0545 - val_accuracy: 0.9931
    Epoch 11/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.0693 - accuracy: 0.9828 - val_loss: 0.0480 - val_accuracy: 0.9917
    Epoch 12/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0283 - accuracy: 0.9893 - val_loss: 0.0551 - val_accuracy: 0.9890
    Epoch 13/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0394 - accuracy: 0.9869 - val_loss: 0.0597 - val_accuracy: 0.9807
    Epoch 14/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.0454 - accuracy: 0.9880 - val_loss: 0.0624 - val_accuracy: 0.9821
    Epoch 15/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0458 - accuracy: 0.9849 - val_loss: 0.0973 - val_accuracy: 0.9642
    Epoch 16/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0720 - accuracy: 0.9807 - val_loss: 0.0651 - val_accuracy: 0.9821
    Epoch 17/40
    91/91 [==============================] - 5s 54ms/step - loss: 0.0454 - accuracy: 0.9862 - val_loss: 0.0673 - val_accuracy: 0.9876
    Epoch 18/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.0411 - accuracy: 0.9859 - val_loss: 0.0936 - val_accuracy: 0.9752
    Epoch 19/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0323 - accuracy: 0.9876 - val_loss: 0.1597 - val_accuracy: 0.9519
    Epoch 20/40
    91/91 [==============================] - 5s 54ms/step - loss: 0.1707 - accuracy: 0.9804 - val_loss: 0.0977 - val_accuracy: 0.9739
    Epoch 21/40
    91/91 [==============================] - 5s 54ms/step - loss: 0.0936 - accuracy: 0.9783 - val_loss: 0.2379 - val_accuracy: 0.9367
    Epoch 22/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.0807 - accuracy: 0.9849 - val_loss: 0.2094 - val_accuracy: 0.9532
    Epoch 23/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.1481 - accuracy: 0.9800 - val_loss: 0.1895 - val_accuracy: 0.9381
    Epoch 24/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.1733 - accuracy: 0.9811 - val_loss: 0.1565 - val_accuracy: 0.9642
    Epoch 25/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.1040 - accuracy: 0.9759 - val_loss: 0.1596 - val_accuracy: 0.9560
    Epoch 26/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.0758 - accuracy: 0.9811 - val_loss: 0.2159 - val_accuracy: 0.9629
    Epoch 27/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.1107 - accuracy: 0.9835 - val_loss: 0.1910 - val_accuracy: 0.9574
    Epoch 28/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0396 - accuracy: 0.9856 - val_loss: 0.2219 - val_accuracy: 0.9436
    Epoch 29/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.1011 - accuracy: 0.9770 - val_loss: 0.2027 - val_accuracy: 0.9422
    Epoch 30/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0321 - accuracy: 0.9893 - val_loss: 1.1109 - val_accuracy: 0.8129
    Epoch 31/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0599 - accuracy: 0.9828 - val_loss: 0.2073 - val_accuracy: 0.9450
    Epoch 32/40
    91/91 [==============================] - 6s 64ms/step - loss: 0.0808 - accuracy: 0.9821 - val_loss: 2.0096 - val_accuracy: 0.7813
    Epoch 33/40
    91/91 [==============================] - 5s 58ms/step - loss: 0.0227 - accuracy: 0.9893 - val_loss: 1.5935 - val_accuracy: 0.7785
    Epoch 34/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.1813 - accuracy: 0.9739 - val_loss: 0.2248 - val_accuracy: 0.9354
    Epoch 35/40
    91/91 [==============================] - 5s 56ms/step - loss: 0.0356 - accuracy: 0.9873 - val_loss: 0.1851 - val_accuracy: 0.9505
    Epoch 36/40
    91/91 [==============================] - 5s 59ms/step - loss: 0.0243 - accuracy: 0.9904 - val_loss: 0.3738 - val_accuracy: 0.9065
    Epoch 37/40
    91/91 [==============================] - 5s 60ms/step - loss: 0.0379 - accuracy: 0.9873 - val_loss: 0.7916 - val_accuracy: 0.8171
    Epoch 38/40
    91/91 [==============================] - 5s 58ms/step - loss: 0.0524 - accuracy: 0.9842 - val_loss: 0.2499 - val_accuracy: 0.9395
    Epoch 39/40
    91/91 [==============================] - 6s 63ms/step - loss: 0.0223 - accuracy: 0.9904 - val_loss: 0.3099 - val_accuracy: 0.9285
    Epoch 40/40
    91/91 [==============================] - 5s 55ms/step - loss: 0.0262 - accuracy: 0.9900 - val_loss: 0.3131 - val_accuracy: 0.9243
    




    <tensorflow.python.keras.callbacks.History at 0x272ecf24370>




```python
Xte_1 = read_image(te.shape[0],200,te)
Xte_2 = Xte_1.reshape(-1,200,200,1)
```

    0 / 2514
    100 / 2514
    200 / 2514
    300 / 2514
    400 / 2514
    500 / 2514
    600 / 2514
    700 / 2514
    800 / 2514
    900 / 2514
    1000 / 2514
    1100 / 2514
    1200 / 2514
    1300 / 2514
    1400 / 2514
    1500 / 2514
    1600 / 2514
    1700 / 2514
    1800 / 2514
    1900 / 2514
    2000 / 2514
    2100 / 2514
    2200 / 2514
    2300 / 2514
    2400 / 2514
    2500 / 2514
    


```python
pred = model.predict(Xte_2)
```


```python
np.round(pred)
```




    array([[1.],
           [0.],
           [0.],
           ...,
           [1.],
           [0.],
           [0.]], dtype=float32)




```python
te1 = te.copy()
```


```python
te1['score'] = pred
```


```python
te1.to_csv('pred1.csv',index = False)
```


```python
te2 = te.copy()
```


```python
te2['score'] = np.round(pred)
```


```python
te2.to_csv('pred2.csv',index = False)
```


```python
te2['score'] = 0
te2.to_csv('pred3.csv',index = False)
te2['score'] = 1
te2.to_csv('pred4.csv',index = False)
```


```python

```
