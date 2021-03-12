```python
import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import backend as Kb
import matplotlib.pyplot as plt
```

# Classifying X-Ray Images With Statistical Learning
---

This notebook/blog documents our building up of a Pipeline and Disease Classifier for the NIH X-Ray Dataset. Our project consisted of analyzing, cleaning, and modifying said dataset for deep learning training, building and training our own classification models, retuning hyperparameters for better accuracy, and comparing results. One of the unique techniques we tested out was training upon the  `discrete latent space` of the `VQ-VAE` model, [*Neural Discrete Representation Learning* (van den Oord et al, NeurIPS 2017)](https://arxiv.org/abs/1906.00446). the Vector-Quantized Variational AutoEncoder (`VQ-VAE`) is a generative model based on Variational Auto Encoders (`VAE`) which aims to make the latent space discrete using Vector Quantization (`VQ`) techniques. We will be comparing the performance of our "hacky" model with a traditional CNN classifier, and the State-of-the-art [Densenet](https://arxiv.org/abs/1608.06993) classifier. Overall, after extensive tuning and refining of our model with statistical optimizations, we found that our preliminary VQ-VAE model performed on par with the Densenet architecture, which in turn, with further optimizations, can perform as well as, or potentially better than, the average board-certified radiologist.

# Dataset and Hyperparameters
First we will load the dataset. Every image is RGB and 1024x1024 pixel. We'll then clean, process, and augment the data so that it's ready to train on.


```python
## Load the NIH data toa pandas dataframe
all_xray_df = pd.read_csv('path/to/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('NIH-XRays','images*', '*', '*.png'))}

print('Total X-Ray images found:', len(all_image_paths))
```

Keeping only Posterior to Anterior (PA) Scans:


```python
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
pa_xray_df = all_xray_df.drop(all_xray_df.loc[all_xray_df['View Position']=='AP'].index)
```

We will now take the diagnosis label, and embed them into a 0-1 vector for every X-Ray Image:


```python
label_set = pa_xray_df.rename(columns={'Finding Labels': 'labels'})
label_set = label_set.set_index('Image Index').labels.str.split('|', expand=True).stack().reset_index(level=1, drop=True).to_frame('labels')
label_set = pd.get_dummies(label_set, columns=['labels']).groupby(level=0).sum()
label_set.head()
```

Merging the original dataframe with the new `label_set` with the labels:


```python
pa_xray_df.set_index('Image Index', inplace=True)
prepared_df = pa_xray_df.merge(label_set, left_index = True, right_index=True)
prepared_df.head()
#prepared_df.to_csv("prepared_df_VAE.csv")
```

    Scans found: 112120 , Total Headers 112120





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
      <th>Finding Labels</th>
      <th>Follow-up #</th>
      <th>Patient ID</th>
      <th>Patient Age</th>
      <th>Patient Gender</th>
      <th>View Position</th>
      <th>OriginalImage[Width</th>
      <th>Height]</th>
      <th>OriginalImagePixelSpacing[x</th>
      <th>y]</th>
      <th>...</th>
      <th>labels_Emphysema</th>
      <th>labels_Fibrosis</th>
      <th>labels_Hernia</th>
      <th>labels_Infiltration</th>
      <th>labels_Mass</th>
      <th>labels_No Finding</th>
      <th>labels_Nodule</th>
      <th>labels_Pleural_Thickening</th>
      <th>labels_Pneumonia</th>
      <th>labels_Pneumothorax</th>
    </tr>
    <tr>
      <th>Image Index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00000001_000.png</th>
      <td>Cardiomegaly</td>
      <td>0</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2682</td>
      <td>2749</td>
      <td>0.143</td>
      <td>0.143</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>00000001_001.png</th>
      <td>Cardiomegaly|Emphysema</td>
      <td>1</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2894</td>
      <td>2729</td>
      <td>0.143</td>
      <td>0.143</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>00000001_002.png</th>
      <td>Cardiomegaly|Effusion</td>
      <td>2</td>
      <td>1</td>
      <td>58</td>
      <td>M</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.168</td>
      <td>0.168</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>00000002_000.png</th>
      <td>No Finding</td>
      <td>0</td>
      <td>2</td>
      <td>81</td>
      <td>M</td>
      <td>PA</td>
      <td>2500</td>
      <td>2048</td>
      <td>0.171</td>
      <td>0.171</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>00000003_000.png</th>
      <td>Hernia</td>
      <td>0</td>
      <td>3</td>
      <td>81</td>
      <td>F</td>
      <td>PA</td>
      <td>2582</td>
      <td>2991</td>
      <td>0.143</td>
      <td>0.143</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
prepared_df = pd.read_csv("prepared_df_VAE.csv", index_col="Image Index")

clean_data, unclean_data = train_test_split(prepared_df, test_size=0.01, random_state=42)
```

    lables_Emphysema 0    65811
    1     1499
    Name: lables_Emphysema, dtype: int64
    lables_Fibrosis 0    65902
    1     1408
    Name: lables_Fibrosis, dtype: int64
    lables_Hernia 0    67118
    1      192
    Name: lables_Hernia, dtype: int64
    lables_Infiltration 0    57957
    1     9353
    Name: lables_Infiltration, dtype: int64
    lables_Mass 0    63743
    1     3567
    Name: lables_Mass, dtype: int64
    lables_No Finding 1    39302
    0    28008
    Name: lables_No Finding, dtype: int64
    lables_Nodule 0    63133
    1     4177
    Name: lables_Nodule, dtype: int64
    lables_Pleural_Thickening 0    64892
    1     2418
    Name: lables_Pleural_Thickening, dtype: int64
    lables_Pneumonia 0    66680
    1      630
    Name: lables_Pneumonia, dtype: int64
    lables_Pneumothorax 0    63903
    1     3407
    Name: lables_Pneumothorax, dtype: int64


Here we are performing image augmentations on the X-Ray images: Centering, resizing, and reformatting the images so that they can be more readily processed and trained on by our models


```python
train_data_augmented = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    featurewise_center=False,
    featurewise_std_normalization=False)


```


```python
labels = ["labels_Emphysema", "labels_Fibrosis", "labels_Hernia", "labels_Infiltration", "labels_Mass", "labels_No Finding", "labels_Nodule", "labels_Pleural_Thickening", "labels_Pneumonia", "labels_Pneumothorax"]
```


```python
train_generator = train_data_augmented.flow_from_dataframe(
    train_data, x_col='path', y_col= labels, weight_col=None,
    target_size=(224, 224), color_mode='rgb', classes=None,
    class_mode='raw', batch_size=66000, shuffle=True, seed=42,
    save_to_dir=None, save_prefix='', save_format='png', subset=None,
    interpolation='nearest', validate_filenames=True)

validation_generator = val_datagen.flow_from_dataframe(
    val_data, x_col='path', y_col=labels, weight_col=None,
    target_size=(224, 224), color_mode='rgb', classes=None,
    class_mode='raw', batch_size=100, shuffle=True, seed=42,
    save_to_dir=None, save_prefix='', save_format='png', subset=None,
    interpolation='nearest', validate_filenames=True)
```

    Found 66636 validated image filenames.


We're going to load up our data now. Fair warning that this will take quite a bit of time!


```python
train_data, train_data_labels = train_generator[0]
```


```python
#np.save("train_data1.npy", train_data1)
single_indices = np.argwhere(np.sum(train_data_labels, axis=1) == 1)
single_train_labels = train_data_labels[single_indices][:,0,:]
single_train_data = train_data[single_indices][:,0,:,:,:]
```

Now that we've cleaned the data so that we are only focusing on single-condition diagnoses, let's see how many of each diagnosis is found in this new cleaned dataset:


```python
dicts = {label : 0 for label in labels}
indices = []
count = 0
for i, arr in enumerate(single_train_labels):
    for j, num in enumerate(arr):
        if j == 5 and num == 1:
            if count < 1000: # Because there is a disproportionately high amount of No Finding X-Rays
                indices.append(i)
                count += 1
            dicts[labels[5]] += 1
            break
            
for i, arr in enumerate(single_train_labels):
    for j, num in enumerate(arr):
        if j != 5 and num == 1:
            indices.append(i)
            dicts[labels[j]] += 1
            break
            
            
# The following code shuffles our fully cleaned data for training purposes
for label in dicts:
    print(label, dicts[label])
indices = np.array(indices)
np.random.shuffle(indices)
train_labels_modified = single_train_labels[indices]
train_data_modified = single_train_data[indices]
```

Here is a sample of what some of the images look like:


```python
t_x, t_y = validation_generator[0]
fig, m_axs = plt.subplots(1, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y[5] == 1: 
        c_ax.set_title('Healthy')
    else:
        c_ax.set_title('Something wrong')
    c_ax.axis('off')
```


![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_20_0.png)


# Building VQ-VAE Architecture 
the VQ-VAE consists of an encoder, a discrete **codebook** of $K$ latent codes, $\mathcal E = \{e_0 \dots e_{K-1} \}$ where $\forall i, e_i \in \mathbb{R}^d$.

\begin{align}
&\textbf{input }x \tag{W x H x C}\\
z_e &= f(x) \tag{w x h x d}\\
z_q^{i, j} &= \arg\min_{e \in \mathcal E} \| z_e^{i, j} - e \|^2
\end{align}

and finally a decoder, that should in theory output a clean reconstruction of the input image.

Below is the forked implementation of the model, with hyperparameters and optimizations adapted for our X-Ray images and current pipeline:


```python
# Hyperparameters
NUM_LATENT_K = 14                  # Number of codebook entries
NUM_LATENT_D = 64                  # Dimension of each codebook entries
BETA = 0.9                         # Weight for the commitment loss

INPUT_SHAPE = (224,224,3)          # Spatial size of latent embedding
SIZE = 14                          
                                   
VQVAE_BATCH_SIZE = 128             # Batch size for training the VQVAE
VQVAE_NUM_EPOCHS = 500             # Number of epochs
VQVAE_LEARNING_RATE = 0.001        # Learning rate
VQVAE_LAYERS = [8, 16, 32, 64]     # Number of filters for each layer in the encoder
```


```python
# Code forked from https://www.kaggle.com/ameroyer/keras-vq-vae-for-image-generation
# Modified and optimized for our pipeline
class VectorQuantizer(K.layers.Layer):  
    def __init__(self, k, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = k
    
    def build(self, input_shape):
        self.d = int(input_shape[-1])
        rand_init = K.initializers.VarianceScaling(distribution="uniform")
        self.codebook = self.add_weight(shape=(self.k, self.d), initializer=rand_init, trainable=True)
        
    def call(self, inputs):
        # Map z_e of shape (b, w,, h, d) to indices in the codebook
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        z_e = tf.expand_dims(inputs, -2)
        dist = tf.norm(z_e - lookup_, axis=-1)
        k_index = tf.argmin(dist, axis=-1)
        return k_index
    
    def sample(self, k_index):
        # Map indices array of shape (b, w, h) to actual codebook z_q
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        k_index_one_hot = tf.one_hot(k_index, self.k)
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q
```

The decoder then takes the quantized codes as inputs and spits out the output image. Our architecture involves a series of convolutional layers that shrink the image by a factor of 2 every layer, followed by a series of transpose convolutional layers that double the image size every layer:


```python
def encoder_pass(inputs, d, num_layers=[8, 16, 32,64]):
    x = inputs

    for i, filters in enumerate(num_layers):
        x = K.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', activation='elu', 
                            kernel_initializer = "he_normal", 
                            strides=(1, 1), name="conv{}".format(i + 1))(x)
        x = K.layers.MaxPooling2D()(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', activation='elu', 
                            kernel_initializer = "he_normal",
                            strides=(1, 1), name="conv2{}".format(i + 1))(x)
        x = K.layers.BatchNormalization()(x)

    z_e = K.layers.Conv2D(filters=d, kernel_size=3, padding='SAME', kernel_initializer = "he_normal",
                          strides=(1, 1), name='z_e')(x)
    return z_e

def decoder_pass(inputs, num_layers=[64,32, 16, 1]):
    y = inputs
    for i, filters in enumerate(num_layers):
        y = K.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(1, 1), padding="SAME", 
                                     kernel_initializer = "he_normal",
                                     activation='elu', name="convT{}".format(i + 1))(y)
        y = K.layers.UpSampling2D()(y)
        y = K.layers.BatchNormalization()(y)
        y = K.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(1, 1), padding="SAME", 
                                     kernel_initializer = "he_normal",
                                     activation='elu', name="convT2{}".format(i + 1))(y)
        y = K.layers.BatchNormalization()(y)
    decoded = K.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=(1, 1), 
                                       kernel_initializer = "he_normal",
                                       padding="SAME", activation='sigmoid', name='output')(y)
    return decoded
```

Once these three building blocks are done, we can build the full `VQ-VAE`. As noted previously, code is forked from online Machine Learning practitioners.


```python
def build_vqvae(k, d, input_shape=(224, 224, 3), num_layers=[8, 16, 32, 64]):
    global SIZE
    ## Encoder
    encoder_inputs = K.layers.Input(shape=input_shape, name='encoder_inputs')
    z_e = encoder_pass(encoder_inputs, d, num_layers=num_layers)
    SIZE = int(z_e.get_shape()[1])

    ## Vector Quantization
    vector_quantizer = VectorQuantizer(k, name="vector_quantizer")
    codebook_indices = vector_quantizer(z_e)
    encoder = K.Model(inputs=encoder_inputs, outputs=codebook_indices, name='encoder')

    ## Decoder
    decoder_inputs = K.layers.Input(shape=(SIZE, SIZE, d), name='decoder_inputs')
    decoded = decoder_pass(decoder_inputs, num_layers=num_layers[::-1])
    decoder = K.Model(inputs=decoder_inputs, outputs=decoded, name='decoder')
    
    ## VQVAE Model (training)
    sampling_layer = K.layers.Lambda(lambda x: vector_quantizer.sample(x), name="sample_from_codebook")
    z_q = sampling_layer(codebook_indices)
    codes = tf.stack([z_e, z_q], axis=-1)
    codes = K.layers.Lambda(lambda x: x, name='latent_codes')(codes)
    straight_through = K.layers.Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator")
    straight_through_zq = straight_through([z_q, z_e])
    reconstructed = decoder(straight_through_zq)
    vq_vae = K.Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')
    
    ## VQVAE model (inference)
    codebook_indices = K.layers.Input(shape=(SIZE, SIZE), name='discrete_codes', dtype=tf.int32)
    z_q = sampling_layer(codebook_indices)
    generated = decoder(z_q)
    vq_vae_sampler = K.Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')
    
    ## Transition from codebook indices to model (for training the prior later)
    indices = K.layers.Input(shape=(SIZE, SIZE), name='codes_sampler_inputs', dtype='int32')
    z_q = sampling_layer(indices)
    codes_sampler = K.Model(inputs=indices, outputs=z_q, name="codes_sampler")
    
    ## Getter to easily access the codebook for vizualisation
    indices = K.layers.Input(shape=(), dtype='int32')
    vector_model = K.Model(inputs=indices, outputs=vector_quantizer.sample(indices[:, None, None]), name='get_codebook')
    def get_vq_vae_codebook():
        codebook = vector_model.predict(np.arange(k))
        codebook = np.reshape(codebook, (k, d))
        return codebook
    
    return vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, get_vq_vae_codebook

```

Alongside the standard MSE Loss, which will be our reconstruction loss, we will also define VQ "Latent" to ensure that encoder outputs stay close to codebook entry values, and also keep the size of the codebook dictionary relatively sparse. the loss function, as defined in the paper, is as follows:

\begin{align}
\mathcal L_{\text{VQ-VAE}}(x) = - \mathbb{E}_{z \sim f(x)}{p(x | z)} + \| z_e - \bar{z_q}\|^2 + \|\bar{z_e} - z_q\|^2
\end{align}

where $\bar{\cdot}$ denotes the stop gradient operation. 

KL Divergence loss is used for training of the prior which is unrelated to our classification problem. This will be an issue once we try to generate random samples, however, which will be discussed later.


```python
def mse_loss(ground_truth, predictions):
    mse_loss = tf.reduce_mean((ground_truth - predictions)**2, name="mse_loss")
    return mse_loss

def latent_loss(dummy_ground_truth, outputs):
    global BETA
    del dummy_ground_truth
    z_e, z_q = tf.split(outputs, 2, axis=-1)
    vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
    commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
    latent_loss = tf.identity(vq_loss + BETA * commit_loss, name="latent_loss")
    return latent_loss    
```



We're also going to monitor the $L2$ [norms](https://mathworld.wolfram.com/VectorNorm.html) of the encoded vectors, both the sample vector, and encoder output vector.


```python
def zq_norm(y_true, y_pred):
    del y_true
    _, z_q = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.norm(z_q, axis=-1))

def ze_norm(y_true, y_pred):
    del y_true
    z_e, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.norm(z_e, axis=-1))
```


```python
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) #check gpu status
early_stop = K.callbacks.EarlyStopping(verbose=1, patience=12, mode='min', restore_best_weights=True)
```

    Num GPUs Available:  8



```python
base_classifier = K.Sequential([
K.layers.Input(shape = (224,224,3)),
K.layers.Conv2D(8, kernel_size = 3, padding = "same", activation = 'elu', kernel_initializer = 'lecun_normal', strides = (1,1)),
K.layers.BatchNormalization(),
K.layers.Conv2D(16, kernel_size = 3, padding = "same", activation = 'elu', kernel_initializer = 'lecun_normal', strides = (1,1)),
K.layers.BatchNormalization(),
K.layers.Conv2D(32, kernel_size = 3, padding = "same", activation = "elu", kernel_initializer = 'lecun_normal', strides = (2,2)),
K.layers.BatchNormalization(),
K.layers.MaxPooling2D(pool_size = (2,2)),
K.layers.Conv2D(64, kernel_size = 3, padding = "same", activation = 'elu', kernel_initializer = "lecun_normal", strides = (2,2)),
K.layers.BatchNormalization(),
K.layers.MaxPooling2D(pool_size = (2,2)),
K.layers.Conv2D(72, kernel_size = 3, padding = "same", activation = "elu", kernel_initializer = "lecun_normal", strides = (1,1)),
K.layers.Conv2D(96, kernel_size = 3, padding = "same", activation = "elu", kernel_initializer = "lecun_normal", strides = (2,2)),
K.layers.BatchNormalization(),
K.layers.Conv2D(10, kernel_size = 3, padding= "same", activation = "elu", kernel_initializer = "lecun_normal", strides = (1,1)),
K.layers.BatchNormalization(),
K.layers.Flatten(),
K.layers.Dense(10, activation = "softmax", kernel_initializer = "lecun_normal") 
])
base_classifier.compile(loss="categorical_crossentropy", metrics = ["accuracy"], optimizer = K.optimizers.Adam(1e-3))
base_classifier.summary()
history_base = base_classifier.fit(train_data_modified, y = train_labels_modified, validation_split=.1, callbacks=[], epochs = 25)
```


```python
densenet = tf.keras.applications.DenseNet121(weights = 'imagenet',
                                             include_top=False, pooling = 'avg',
                                             input_shape=[224, 224, 3])
densenet.trainable = True
dense_model = tf.keras.Sequential([
    densenet,
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer = "lecun_normal")
])
dense_model.compile(loss="categorical_crossentropy", metrics = ["accuracy"], optimizer = K.optimizers.Adam(1e-3))
dense_model.summary()
history_dense = dense_model.fit(train_data_modified, y = train_labels_modified, validation_split = 0.2, callbacks=[early_stop], epochs = 100)
```


```python
vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, get_vq_vae_codebook = build_vqvae(NUM_LATENT_K, NUM_LATENT_D, input_shape=INPUT_SHAPE, num_layers=VQVAE_LAYERS)
vq_vae.compile(loss=[mse_loss, latent_loss], metrics={"latent_codes": [zq_norm, ze_norm]}, optimizer= K.optimizers.Adam(1e-4, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False))
vq_vae.summary()
```

    WARNING:tensorflow:Entity <bound method VectorQuantizer.call of <__main__.VectorQuantizer object at 0x7ef0c74d6d30>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method VectorQuantizer.call of <__main__.VectorQuantizer object at 0x7ef0c74d6d30>>: AssertionError: Bad argument number for Name: 3, expecting 4
    WARNING: Entity <bound method VectorQuantizer.call of <__main__.VectorQuantizer object at 0x7ef0c74d6d30>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method VectorQuantizer.call of <__main__.VectorQuantizer object at 0x7ef0c74d6d30>>: AssertionError: Bad argument number for Name: 3, expecting 4
    Model: "vq-vae"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    encoder_inputs (InputLayer)     [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    conv1 (Conv2D)                  (None, 224, 224, 8)  224         encoder_inputs[0][0]             
    __________________________________________________________________________________________________
    max_pooling2d_101 (MaxPooling2D (None, 112, 112, 8)  0           conv1[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_1026 (Batch (None, 112, 112, 8)  32          max_pooling2d_101[0][0]          
    __________________________________________________________________________________________________
    conv21 (Conv2D)                 (None, 112, 112, 8)  584         batch_normalization_1026[0][0]   
    __________________________________________________________________________________________________
    batch_normalization_1027 (Batch (None, 112, 112, 8)  32          conv21[0][0]                     
    __________________________________________________________________________________________________
    conv2 (Conv2D)                  (None, 112, 112, 16) 1168        batch_normalization_1027[0][0]   
    __________________________________________________________________________________________________
    max_pooling2d_102 (MaxPooling2D (None, 56, 56, 16)   0           conv2[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_1028 (Batch (None, 56, 56, 16)   64          max_pooling2d_102[0][0]          
    __________________________________________________________________________________________________
    conv22 (Conv2D)                 (None, 56, 56, 16)   2320        batch_normalization_1028[0][0]   
    __________________________________________________________________________________________________
    batch_normalization_1029 (Batch (None, 56, 56, 16)   64          conv22[0][0]                     
    __________________________________________________________________________________________________
    conv3 (Conv2D)                  (None, 56, 56, 32)   4640        batch_normalization_1029[0][0]   
    __________________________________________________________________________________________________
    max_pooling2d_103 (MaxPooling2D (None, 28, 28, 32)   0           conv3[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_1030 (Batch (None, 28, 28, 32)   128         max_pooling2d_103[0][0]          
    __________________________________________________________________________________________________
    conv23 (Conv2D)                 (None, 28, 28, 32)   9248        batch_normalization_1030[0][0]   
    __________________________________________________________________________________________________
    batch_normalization_1031 (Batch (None, 28, 28, 32)   128         conv23[0][0]                     
    __________________________________________________________________________________________________
    conv4 (Conv2D)                  (None, 28, 28, 64)   18496       batch_normalization_1031[0][0]   
    __________________________________________________________________________________________________
    max_pooling2d_104 (MaxPooling2D (None, 14, 14, 64)   0           conv4[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_1032 (Batch (None, 14, 14, 64)   256         max_pooling2d_104[0][0]          
    __________________________________________________________________________________________________
    conv24 (Conv2D)                 (None, 14, 14, 64)   36928       batch_normalization_1032[0][0]   
    __________________________________________________________________________________________________
    batch_normalization_1033 (Batch (None, 14, 14, 64)   256         conv24[0][0]                     
    __________________________________________________________________________________________________
    z_e (Conv2D)                    (None, 14, 14, 64)   36928       batch_normalization_1033[0][0]   
    __________________________________________________________________________________________________
    vector_quantizer (VectorQuantiz (None, 14, 14)       896         z_e[0][0]                        
    __________________________________________________________________________________________________
    sample_from_codebook (Lambda)   (None, 14, 14, 64)   0           vector_quantizer[0][0]           
    __________________________________________________________________________________________________
    straight_through_estimator (Lam (None, 14, 14, 64)   0           sample_from_codebook[0][0]       
                                                                     z_e[0][0]                        
    __________________________________________________________________________________________________
    tf_op_layer_stack_64 (TensorFlo [(None, 14, 14, 64,  0           z_e[0][0]                        
                                                                     sample_from_codebook[0][0]       
    __________________________________________________________________________________________________
    decoder (Model)                 (None, 224, 224, 3)  197003      straight_through_estimator[0][0] 
    __________________________________________________________________________________________________
    latent_codes (Lambda)           (None, 14, 14, 64, 2 0           tf_op_layer_stack_64[0][0]       
    ==================================================================================================
    Total params: 309,395
    Trainable params: 308,435
    Non-trainable params: 960
    __________________________________________________________________________________________________



```python
history_vq = vq_vae.fit(single_train_data, y = [single_train_data, None], validation_split=.2, callbacks= [early_stop], epochs=100, verbose=2)
vq_vae.save_weights("FINAL_TRAINED_VQ.h5")
'''
Current Stats after 69 Epochs:
loss: 0.0054 - decoder_loss: 0.0023 - latent_codes_loss: 0.0032
latent_codes_zq_norm: 0.1514 - latent_codes_ze_norm: 0.1544
val_loss: 0.0056 - val_decoder_loss: 0.0023 - val_latent_codes_loss: 0.0032
val_latent_codes_zq_norm: 0.1503 - val_latent_codes_ze_norm: 0.1518
'''
```

Now that our VQ-VAE model is trained, let's look at the loss charts to see when or if overfitting occurred.


```python
num_epochs = range(1, len(history_vq.history["loss"]) + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(num_epochs, history_vq.history["loss"], label="total loss")
plt.plot(num_epochs, history_vq.history["decoder_loss"], label="reconstruction loss")
plt.plot(num_epochs, history_vq.history["latent_codes_loss"], label="codebook loss")
plt.plot(num_epochs, history_vq.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss during training")
plt.legend()
plt.title("Losses")
plt.subplot(1, 2, 2)
plt.plot(num_epochs, history_vq.history["latent_codes_ze_norm"], label="ze_norm")
plt.plot(num_epochs, history_vq.history["latent_codes_zq_norm"], label="zq_norm")
plt.xlabel("Epoch")
plt.ylabel("Norm")
plt.title("Codes norms during training")
plt.legend()
plt.show()
```


![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_38_0.png)


Let's check out what some of the reconstructions look like from this VQ-VAE. Note that since this is an X-Ray image, we are most likely not able to get perfect reconstructions, but instead we should look for promising results.


```python
def reconstruction(model, images, n_row, n_col, random=True):
    n = n_row * n_col
    if random:
        x = np.random.choice(images.shape[0], size=n, replace=False)
        x = images[x]
    else:
        x = images[:n]
    recons, code = model.predict(x)
    dists = np.mean((recons - x)**2, axis=(1, 2, 3))
    plt.figure(figsize=(15, 9))
    for i in range(n):
        plt.subplot(n_row, 2 * n_col, 2 * i + 1)
        plt.imshow(x[i, :, :, 1], cmap='gray')
        plt.title("original", fontsize=7)
        plt.axis('off')
        plt.subplot(n_row, 2 * n_col, 2 * i + 2)
        plt.imshow(recons[i, :, :, 2], cmap='gray')
        plt.title("L2: {:.3f}".format(dists[i]), fontsize=7)
        plt.axis('off')
    plt.show()
    
print("Left: Original, Right: Generated reconstruction")
reconstruction(vq_vae, train_data_modified, 1, 3, random = True)
```

    Originals versus reconstructions



![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_40_1.png)


These reconstructions are actually quite decent! Now, our hypothesis is that the latent space representation might have "learned" some information about the images that's valuable for diagnosing the conditions. Let's see whether or not this is true:


```python
#vq_vae.load_weights("FINAL_TRAINED_VQ.h5")
latent = encoder.predict(train_data_modified)
classifier = K.Sequential([K.layers.Input(shape = latent.shape[1:]),
                           K.layers.Flatten(),
                           K.layers.BatchNormalization(),
                           K.layers.Dense(128, activation = 'selu', kernel_initializer = 'lecun_normal'),
                           K.layers.BatchNormalization(),
                           K.layers.Dense(64, activation = 'selu', kernel_initializer = 'lecun_normal'),
                           K.layers.BatchNormalization(),
                           K.layers.Dense(10, activation = "softmax", kernel_initializer = 'glorot_normal')])

classifier.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer = K.optimizers.Adam(1e-3))
classifier.summary()
history_vq_class = classifier.fit(latent, y = train_labels_modified, callbacks=[early_stop], epochs = 100)
```

Here is a simple classifier for single images from our test data:


```python
def make_prediction(encoder, classifier, image):
    global labels
    image = np.expand_dims(image, axis = 0)
    latent = encoder.predict(image)
    output = classifier.predict(latent)
    label = labels[np.argmax(output)]
    return "This X-Ray diagnosis: " + label[7:]

make_prediction(encoder, classifier, train_data_modified[1000])
```




    'This X-Ray diagnosis: Infiltration'



Finally, we have 3 trained models, and have measured and saved their performances. Let's plot the graphs to see how well each of them did!


```python
loss_base = history_base.history["loss"]
acc_base = history_base.history["acc"]
base_best = np.argmin(history_base.history["val_loss"])[-1]

loss_dense = history_dense.history["loss"]
acc_dense = history_dense.history["acc"]
dense_best = np.argmin(history_dense.history["val_loss"])[-1]

loss_vq = history_vq_class.history["loss"]
acc_vq = history_vq_class.history["acc"]
vq_best = np.argmin(history_vq_class.history["val_loss"])[-1]


num_epochs = range(1, 37)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, 26), loss_vq[:25], label="VQ-VAE loss")
plt.plot(np.arange(1,16), history_dense.history['loss'][:15], label="Densenet loss")
plt.plot(np.arange(1,9), history_base.history['loss'][:8], label="Simple CNN loss")

plt.xlabel("Epoch")
plt.ylabel("Loss before overfitting")
plt.legend()
plt.title("Losses")

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, 26), acc_vq[:25], label="VQ-VAE accuracy")
plt.plot(np.arange(1,16), history_dense.history['acc'][:15], label="Densenet accuracy")
plt.plot(np.arange(1,9), history_base.history['acc'][:8], label="Simple CNN accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy before overfitting")
plt.title("Accuracies")
plt.legend()
#plt.show()
```


![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_46_0.png)


As one can see, the simple CNN fares the worst, whereas the VQ-VAE and the Densenet Architecture classifications are on par with each other, which is quite exciting for our findings. While the accuracy ratings top out at around 60% for each on our multiclass classification tests, we believe that with further hyperparameter tuning and layer reshaping, our models can perform up to par with board-certified radiologists, and possibly surpass them.

# X-Ray Generation
Another stretch goal of ours we had was to use the VQ-VAE as a generator for new X-Ray images. As the architecture currently stands, images generated from a random sample from the latent space are highly unlikely to be resemblant of actual X-Ray images due to the prior being assumed as a uniform, or sometimes basic Gaussian N(0,1) distribution. The following images are then generated from sampling from said distributions:


```python
PIXELCNN_NUM_EPOCHS = 10          # Number of epochs
PIXELCNN_LEARNING_RATE = 3e-4     # Learning rate
PIXELCNN_NUM_BLOCKS = 12          # Number of Gated PixelCNN blocks in the architecture
PIXELCNN_NUM_FEATURE_MAPS = 48    # Width of each PixelCNN block

def generate_from_random(sampler, k, size, n_row, n_col):
    n = n_col * n_row
    indices = np.random.randint(0, k, size=(n, size, size))
    generated = sampler.predict(indices, steps=1)
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(generated[i,:,:,0], cmap='gray')
        plt.axis('off')
    plt.show()
    
print("Images generated under a uniform prior:")
generate_from_random(vq_vae_sampler, NUM_LATENT_K, SIZE, 3, 3)
```

    Images generated under a uniform prior (training assumption)



![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_49_1.png)


As noted in the VQ-VAE2 paper, we decided to make use of the PixelSnail CNN architecture to train on the latent space in order to generate more realistic "random" latent representations, which would in turn be converted into an entirely new X-Ray image (with a diagnosis from our classifier model) via our trained VQ-VAE decoder. Below is the code for the CNN Architecture, courtesy of some amazing ML practitioners online:


```python
# Referencing https://www.kaggle.com/ameroyer/keras-vq-vae-for-image-generation

def gate(inputs):
    """Gated activations"""
    x, y = tf.split(inputs, 2, axis=-1)
    return Kb.tanh(x) * Kb.sigmoid(y)


class MaskedConv2D(K.layers.Layer):
    """Masked convolution"""
    def __init__(self, kernel_size, out_dim, direction, mode, **kwargs):
        self.direction = direction     # Horizontal or vertical
        self.mode = mode               # Mask type "a" or "b"
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        super(MaskedConv2D, self).__init__(**kwargs)
    
    def build(self, input_shape):   
        filter_mid_y = self.kernel_size[0] // 2
        filter_mid_x = self.kernel_size[1] // 2        
        in_dim = int(input_shape[-1])
        w_shape = [self.kernel_size[0], self.kernel_size[1], in_dim, self.out_dim]
        mask_filter = np.ones(w_shape, dtype=np.float32)
        # Build the mask
        if self.direction == "h":
            mask_filter[filter_mid_y + 1:, :, :, :] = 0.
            mask_filter[filter_mid_y, filter_mid_x + 1:, :, :] = 0.
        elif self.direction == "v":
            if self.mode == 'a':
                mask_filter[filter_mid_y:, :, :, :] = 0.
            elif self.mode == 'b':
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
        if self.mode == 'a':
            mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        # Create convolution layer parameters with masked kernel
        self.W = mask_filter * self.add_weight("W_{}".format(self.direction), w_shape, trainable=True)
        self.b = self.add_weight("v_b", [self.out_dim,], trainable=True)
    
    def call(self, inputs):
        return K.backend.conv2d(inputs, self.W, strides=(1, 1)) + self.b

    
def gated_masked_conv2d(v_stack_in, h_stack_in, out_dim, kernel, mask='b', residual=True, i=0):
    """Basic Gated-PixelCNN block. 
       This is an improvement over PixelRNN to avoid "blind spots", i.e. pixels missingt from the
       field of view. It works by having two parallel stacks, for the vertical and horizontal direction, 
       each being masked  to only see the appropriate context pixels.
    """
    kernel_size = (kernel // 2 + 1, kernel)
    padding = (kernel // 2, kernel // 2)
        
    v_stack = K.layers.ZeroPadding2D(padding=padding, name="v_pad_{}".format(i))(v_stack_in)
    v_stack = MaskedConv2D(kernel_size, out_dim * 2, "v", mask, name="v_masked_conv_{}".format(i))(v_stack)
    v_stack = v_stack[:, :int(v_stack_in.get_shape()[-3]), :, :]
    v_stack_out = K.layers.Lambda(lambda inputs: gate(inputs), name="v_gate_{}".format(i))(v_stack)
    
    kernel_size = (1, kernel // 2 + 1)
    padding = (0, kernel // 2)
    h_stack = K.layers.ZeroPadding2D(padding=padding, name="h_pad_{}".format(i))(h_stack_in)
    h_stack = MaskedConv2D(kernel_size, out_dim * 2, "h", mask, name="h_masked_conv_{}".format(i))(h_stack)
    h_stack = h_stack[:, :, :int(h_stack_in.get_shape()[-2]), :]
    h_stack_1 = K.layers.Conv2D(filters=out_dim * 2, kernel_size=1, strides=(1, 1), name="v_to_h_{}".format(i))(v_stack)
    h_stack_out = K.layers.Lambda(lambda inputs: gate(inputs), name="h_gate_{}".format(i))(h_stack + h_stack_1)
    
    h_stack_out =  K.layers.Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), name="res_conv_{}".format(i))(h_stack_out)
    if residual:
        h_stack_out += h_stack_in
    return v_stack_out, h_stack_out


def accuracy(y_true, y_pred):
    size = int(y_pred.get_shape()[-2])
    k = int(y_pred.get_shape()[-1])
    y_true = tf.reshape(y_true, (-1, size * size))
    y_pred = tf.reshape(y_pred, (-1, size * size, k))
    return Kb.cast(Kb.equal(y_true, Kb.cast(Kb.argmax(y_pred, axis=-1), Kb.floatx())), Kb.floatx())


# Now to build the PixelCNN:

def build_pixelcnn(codes_sampler, k, size, num_layers, num_feature_maps=32):
    pixelcnn_prior_inputs = K.layers.Input(shape=(size, size), name='pixelcnn_prior_inputs', dtype=tf.int32)
    z_q = codes_sampler(pixelcnn_prior_inputs) # maps indices to the actual codebook
    
    v_stack_in, h_stack_in = z_q, z_q
    for i in range(num_layers):
        mask = 'b' if i > 0 else 'a'
        kernel_size = 3 if i > 0 else 7
        residual = True if i > 0 else False
        v_stack_in, h_stack_in = gated_masked_conv2d(v_stack_in, h_stack_in, num_feature_maps,
                                                     kernel=kernel_size, residual=residual, i=i + 1)

    fc1 = K.layers.Conv2D(filters=num_feature_maps, kernel_size=1, name="fc1")(h_stack_in)
    fc2 = K.layers.Conv2D(filters=k, kernel_size=1, name="fc2")(fc1) 
    # outputs logits for probabilities of codebook indices for each cell

    pixelcnn_prior = K.Model(inputs=pixelcnn_prior_inputs, outputs=fc2, name='pixelcnn-prior')

    # Distribution to sample from the pixelcnn
    dist = tf.distributions.Categorical(logits=fc2)
    sampled = dist.sample()
    prior_sampler = K.Model(inputs=pixelcnn_prior_inputs, outputs=sampled, name='pixelcnn-prior-sampler')
    return pixelcnn_prior, prior_sampler

pixelcnn_prior, prior_sampler = build_pixelcnn(codes_sampler, NUM_LATENT_K, SIZE, 
                                               PIXELCNN_NUM_BLOCKS, PIXELCNN_NUM_FEATURE_MAPS)
pixelcnn_prior.summary()
```

In order to get a better understanding of our latent space, here's a way to visualize the discrete codes representation as a mapping to a color:


```python
def viz_codes(encoder, images, n_row, n_col, random=True):
    n = n_row * n_col
    if random:
        x = np.random.choice(images.shape[0], size=n, replace=False)
        x = images[x]
    else:
        x = images[:n]
    codes = encoder.predict(x)
    fig = plt.figure(figsize=(15, 8))
    for i in range(n):
        plt.subplot(n_row, 2 * n_col, 2*i+1)
        plt.imshow(x[i,:,:,0], cmap='gray')
        plt.axis('off')
        plt.subplot(n_row, 2 * n_col, 2*i+2)
        plt.pcolor(codes[i, ::-1, :], cmap='tab10') #flip y-axis origin to match imshow
        plt.axis('off')
    cbaxes = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    cb = plt.colorbar(cax=cbaxes)  
    plt.show()
    
print("Discrete latent space as an array: \r\n", latent[0])
print("Images encoded as discrete codes")
viz_codes(encoder, single_train_data[:100], 2, 2)
```

    Discrete latent space as an array: 
     [[ 3  3  3  3 12  5  8  5 12  3  3  3  3  3]
     [12 12 12 12  5  5  5  5  5 12 10 10 10 12]
     [ 5  5  5  5  5  5  5  5  5  5  5  5  5  8]
     [ 8  5  5  5 12  5  5 12 12  5  5  5  5  8]
     [ 5  5  5 12 12  5  5  5 12 12  5  5  5  5]
     [ 5  5  8 12  5  5  5  5 12 12  5  5  5  5]
     [ 8  5  8 12  5  5  5  5  5 12  5  5  5  5]
     [ 5  5  5  5  5  5  8  5  5  5  5  8  5  5]
     [ 8  8  5  8  8  8  8  8 12  5 12  8  5  5]
     [ 8  8  5  5  5  8  8  5  5  8  5  5  8  8]
     [ 8  8  5  8  8  8  8  8  8  8  8  8  5  5]
     [ 8  8  8  8  8  8  8  8  8  8  8  8  8  8]
     [ 8  8  8  8  8  8  8  8  8  8  8  8  8 12]
     [ 8  8  8  8  8  8  8  8  8  8  8  8  5  5]]
    Images encoded as discrete codes



![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_53_1.png)


Now to train our CNN to develop new latent samples:


```python
pixelcnn_prior.compile(loss=K.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[accuracy],
                       optimizer=K.optimizers.Adam(PIXELCNN_LEARNING_RATE))

prior_history = pixelcnn_prior.fit(latent, latent, callbacks=[early_stop], epochs=60, verbose=1)
```


```python
num_epochs = range(1, len(prior_history.history["loss"]) + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(num_epochs, prior_history.history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss during training")
plt.subplot(1, 2, 2)
plt.plot(num_epochs, prior_history.history["accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy during training")
plt.show()
```


![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_56_0.png)


Once again, we can check the model ability to **reconstruct** discrete latent codes obtained from the test set:


```python
def recons_prior(model, z, n_row, n_col, random=True):
    n = n_row * n_col
    if random:
        x = np.random.choice(z.shape[0], size=n, replace=False)
        x = z[x]
    else:
        x = z[:n]
    recons = model.predict(x)
    recons = np.argmax(recons, axis=-1)
    accs = np.mean(np.equal(recons, x), axis=(1, 2))
    plt.figure(figsize=(15, 9))
    for i in range(n):
        plt.subplot(n_row, 2 * n_col, 2 * i + 1)
        plt.imshow(x[i], cmap='tab10')
        plt.title("original", fontsize=8)
        plt.axis('off')
        plt.subplot(n_row, 2 * n_col, 2 * i + 2)
        plt.imshow(recons[i], cmap='tab10')
        plt.title("acc: {:.3f}".format(accs[i]), fontsize=8)
        plt.axis('off')
    plt.show()
    
print("Originals versus reconstructions")
z_test = encoder.predict(latent[1000:1100])
recons_prior(pixelcnn_prior, z_test, 1, 4)
```

    Originals versus reconstructions



![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_58_1.png)


Now when we sample from this new "prior" PixelCNN and run it through the VQ-VAE decoder, we get completely new X-Ray images!


```python
def sample_from_prior(prior, shape):
    """sample from the PixelCNN prior, pixel by pixel"""
    global prior_sampler
    X = np.zeros(shape, dtype=np.int32)
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            sampled = prior_sampler.predict(X)
            X[:, i, j] = sampled[:, i, j]
    return X
            
    
def generate(prior, codes_sampler, size, n_row, n_col):
    """Generate random images by sampling codes from the given prior"""
    n = n_col * n_row
    indices = sample_from_prior(prior, (n, size, size))
    zq = codes_sampler(indices)
    generated = decoder.predict(zq, steps=1)
    plt.figure(figsize=(15, 8))
    for i in range(n):
        plt.subplot(n_row, 2 * n_col, 2 * i + 1)
        plt.imshow(indices[i], cmap='tab10')
        plt.axis('off')
        plt.subplot(n_row, 2 * n_col, 2 * i + 2)
        plt.imshow(generated[i,:,:,0], cmap='gray')
        plt.axis('off')
    plt.show()
    
print("Random codes sampled from the prior and corresponding decoded images")
generate(prior_sampler, codes_sampler, 14, 1, 3)
```

    Random codes sampled from the prior and corresponding decoded images



![png](X-Ray%20Classification%20Pipeline_files/X-Ray%20Classification%20Pipeline_60_1.png)

