# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 23:20:32 2023

@author: johna

Working a ViT concept in 3D. Assume input size is 40,128,128
"""

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# first, prepare the patch extraction operation

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size' : self.patch_size
            })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.extract_volume_patches(
            images,
            ksizes=[1, *self.patch_size, 1],
            strides=[1, *self.patch_size, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches' : self.num_patches,
            'projection_dim' : self.projection_dim
            })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def _early_fusion_direct(filters, mode='concat'):
    def f(in_tensor,fuseto):
        while in_tensor.shape[-1] != 16:
            in_tensor = layers.Dense(units=max(round(in_tensor.shape[-1]/2),16),
                              activation='relu', kernel_initializer='he_normal')(in_tensor)
            in_tensor = layers.Dropout(0.2)(in_tensor)
        to3d = layers.Reshape((1,1,1,-1))(in_tensor)
        dest_shape = fuseto.shape[1:-1]
        extended = layers.UpSampling3D(size=dest_shape)(to3d)
        if mode == 'concat':
            extended = layers.Concatenate(axis=-1)([fuseto,extended])
        elif mode == 'add':
            extended = layers.Add()([fuseto,extended])
        return extended
    return f

def _early_fusion_conv(filters, reg_factor, mode='concat'):
    def f(in_tensor,fuseto):
        while in_tensor.shape[-1] != 16:
            in_tensor = layers.Dense(units=max(round(in_tensor.shape[-1]/2),16),
                              activation='relu', kernel_initializer='he_normal')(in_tensor)
            in_tensor = layers.Dropout(0.2)(in_tensor)
        to3d = layers.Reshape((1,1,1,-1))(in_tensor)
        to3d = layers.UpSampling3D(size=(5,4,4))(to3d)
        while to3d.shape[2] < fuseto.shape[2]:
            if to3d.shape[1] == fuseto.shape[1]:
                upsize = (1,2,2)
            else:
                upsize = (2,2,2)
            to3d = layers.Conv3D(
                filters, (3,3,3), strides=(1,1,1), padding='same',
                kernel_initializer='he_normal'
                )(to3d)
            to3d = layers.Conv3D(
                filters, (3,3,3), strides=(1,1,1), padding='same',
                kernel_initializer='he_normal'
                )(to3d)
            to3d = layers.UpSampling3D(size=upsize)(to3d)
        if mode == 'concat':
            extended = layers.Concatenate(axis=-1)([fuseto,to3d])
        elif mode == 'add':
            extended = layers.Add()([fuseto,to3d])
        return extended
    return f

def ViTBuilder(
        input_shape,
        basefilters=64,
        transformer_layers=8,
        num_heads=4,
        mlp_head_units=[512,256],
        num_outputs=1,
        fusions={}
        ):
    patch_size = (
        int(input_shape[0] / 5),
        int(input_shape[1] / (2**3)),
        int(input_shape[2] / (2**3))
        )
    print("Patch size:", patch_size)
    projection_dim = basefilters #convenience rename
    num_patches = int(math.prod(
        [(input_shape[x] / patch_size[x]) for x in range(3)]
        ))
    transformer_units = [
        projection_dim * 2,
        projection_dim
        ]
    inputs = layers.Input(shape=input_shape)
    if 0 in fusions.keys():
        if not isinstance(inputs,list):
            inputs = [inputs]
        init_shape = fusions[0]
        newinputs = keras.Input(shape=(init_shape,))
        inputs.append(newinputs)
        
        if fusions.get('concat',False):
            mode = 'concat'
        elif fusions.get('add',True):
            mode = 'add'
        
        if fusions.get('direct',False):
            prepatches = _early_fusion_direct(basefilters,mode=mode)(
                newinputs,inputs[0]
                )
        elif fusions.get('conv',True):
            prepatches = _early_fusion_conv(basefilters,mode=mode)(
                newinputs,inputs[0]
                )
    else:
        prepatches = inputs
    
    # Create patches.
    patches = Patches(patch_size)(prepatches)
    # Encode patches.
    print("Number of patches:",num_patches)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    if 'late' in fusions.keys():
        if not isinstance(inputs,list):
            inputs = [inputs]
        newinputs = keras.Input(shape=(fusions['late'],))
        inputs.append(newinputs)
        in_tensor = newinputs # quick convenience rename
        while in_tensor.shape[-1] != 16:
            in_tensor = layers.Dense(units=max(round(in_tensor.shape[-1]/2),16),
                              activation='relu', kernel_initializer='he_normal')(in_tensor)
        features = layers.Concatenate()([features,in_tensor])
        features = layers.Dense(units=16,activation='relu',
                         kernel_initializer='he_normal')(features)
    
    # Classify outputs.
    logits = layers.Dense(num_outputs,activation='sigmoid')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model