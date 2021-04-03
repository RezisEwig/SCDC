import tensorflow as tf
import keras

 
def Deep_Model1():
    input = keras.layers.Input(shape=(226,))
    h1 = keras.layers.Dense(226, activation='sigmoid')(input)
    h2 = keras.layers.Dense(226, activation='sigmoid')(h1)
    h3 = keras.layers.Dense(226, activation='sigmoid')(h2)
    h4 = keras.layers.Dense(113, activation='sigmoid')(h3)
    h5 = keras.layers.Dense(113, activation='sigmoid')(h4)
    added = keras.layers.Add()([h4, h5])
    h6 = keras.layers.Dense(113, activation='sigmoid')(added)
    added2 = keras.layers.Add()([h4, h6])
    h7 = keras.layers.Dense(113, activation='sigmoid')(added2)
    added3 = keras.layers.Add()([h4, h7])
    h8 = keras.layers.Dense(113, activation='sigmoid')(added3)
    h9 = keras.layers.Dense(113, activation='sigmoid')(h8)
    out = keras.layers.Dense(11, activation='sigmoid')(h9)

    model = keras.models.Model(inputs=[input], outputs=out)
       
    return model


def Deep_Model2():
    input = keras.layers.Input(shape=(226,))
    h1 = keras.layers.Dense(226, activation='sigmoid')(input)
    h2 = keras.layers.Dense(226, activation='sigmoid')(h1)
    h3 = keras.layers.Dense(226, activation='sigmoid')(h2)
    out = keras.layers.Dense(11, activation='sigmoid')(h3)

    model = keras.models.Model(inputs=[input], outputs=out)
        
    return model