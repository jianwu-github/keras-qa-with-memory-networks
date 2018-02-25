## Training End-To-End Memory Networks for Question Answering Tasks with Kears

This project is based on the [Keras End-To-End Memory Networks example](https://github.com/keras-team/keras/blob/master/examples/babi_memnn.py) by Francois Chollet

To train End-To-End Memory Network Model using a subset of Facebook bAbI dataset, run train_memnn_with_babi_10k.py through command line:

```bash
$ python3 train_memnn_with_babi_10k.py -bs 32 -ep 120
/usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
/usr/local/lib/python3.6/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
Train on 10000 samples, validate on 1000 samples
Epoch 1/120
2018-02-25 11:33:44.229761: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
10000/10000 [==============================] - 4s 443us/step - loss: 1.9695 - acc: 0.1628 - val_loss: 1.7957 - val_acc: 0.1710
Epoch 2/120
10000/10000 [==============================] - 4s 401us/step - loss: 1.7480 - acc: 0.2311 - val_loss: 1.6925 - val_acc: 0.2630
Epoch 3/120
10000/10000 [==============================] - 4s 391us/step - loss: 1.6706 - acc: 0.2788 - val_loss: 1.6110 - val_acc: 0.2770
Epoch 4/120
10000/10000 [==============================] - 4s 400us/step - loss: 1.5938 - acc: 0.3540 - val_loss: 1.4887 - val_acc: 0.4090
Epoch 5/120
10000/10000 [==============================] - 4s 381us/step - loss: 1.5077 - acc: 0.3929 - val_loss: 1.4597 - val_acc: 0.4120
Epoch 6/120
10000/10000 [==============================] - 4s 397us/step - loss: 1.4906 - acc: 0.3990 - val_loss: 1.4670 - val_acc: 0.4220
Epoch 7/120
10000/10000 [==============================] - 4s 402us/step - loss: 1.4640 - acc: 0.4233 - val_loss: 1.4418 - val_acc: 0.4600
... ...
10000/10000 [==============================] - 3s 289us/step - loss: 0.0729 - acc: 0.9751 - val_loss: 0.1269 - val_acc: 0.9640
Epoch 114/120
10000/10000 [==============================] - 3s 293us/step - loss: 0.0721 - acc: 0.9752 - val_loss: 0.1164 - val_acc: 0.9680
Epoch 115/120
10000/10000 [==============================] - 3s 293us/step - loss: 0.0644 - acc: 0.9776 - val_loss: 0.1268 - val_acc: 0.9640
Epoch 116/120
10000/10000 [==============================] - 3s 288us/step - loss: 0.0651 - acc: 0.9759 - val_loss: 0.1337 - val_acc: 0.9560
Epoch 117/120
10000/10000 [==============================] - 3s 289us/step - loss: 0.0664 - acc: 0.9776 - val_loss: 0.1394 - val_acc: 0.9630
Epoch 118/120
10000/10000 [==============================] - 3s 288us/step - loss: 0.0684 - acc: 0.9786 - val_loss: 0.1671 - val_acc: 0.9470
Epoch 119/120
10000/10000 [==============================] - 3s 304us/step - loss: 0.0630 - acc: 0.9793 - val_loss: 0.1344 - val_acc: 0.9660
Epoch 120/120
10000/10000 [==============================] - 3s 319us/step - loss: 0.0623 - acc: 0.9785 - val_loss: 0.1195 - val_acc: 0.9710
```


To test trained Keras End-To-End Memory Network Model with sample story and question, run qa_with_memnn_model.py through command line:

```bash
$ python3 qa_with_memnn_model.py 
/usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
/usr/local/lib/python3.6/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
2018-02-24 01:12:14.613539: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Story Vocabularies: ['.', '?', 'Daniel', 'John', 'Mary', 'Sandra', 'Where', 'back', 'bathroom', 'bedroom', 'garden', 'hallway', 'is', 'journeyed', 'kitchen', 'moved', 'office', 'the', 'to', 'travelled', 'went'] 

Test Story: Sandra went to the hallway . John journeyed to the bathroom . Sandra travelled to the office . 

Test Question: Where is Sandra ? 

Predicted Answer: office([17]) 

```
