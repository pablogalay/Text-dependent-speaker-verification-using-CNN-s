import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import Model, layers # type: ignore
from PreprocesamientoDatos import get_mel_spectrogram

FRECUENCIA_MUESTREO = 16000
DOS_SEGUNDOS = 2
MUESTRAS_2S = (FRECUENCIA_MUESTREO*DOS_SEGUNDOS)

# Crear una capa personalizada para aplicar get_mel_spectrogram
class MelSpectrogramLayer(layers.Layer):
    def __init__(self, n_mel_bins=40, **kwargs):
        super(MelSpectrogramLayer, self).__init__(**kwargs)
        self.n_mel_bins = n_mel_bins
        
    def call(self, inputs):
        # Redimensionar los datos de entrada
        waveforms = tf.reshape(inputs, [-1, MUESTRAS_2S])
        
        # Procesar cada forma de onda para obtener su espectrograma
        spectrograms = tf.map_fn(
            lambda x: get_mel_spectrogram(x, self.n_mel_bins),
            waveforms,
            fn_output_signature=tf.float32
        )
        
        return spectrograms
    
    def get_config(self):
        config = super(MelSpectrogramLayer, self).get_config()
        config.update({'n_mel_bins': self.n_mel_bins})
        return config

def Modelo_CNN_Frases(formatoEntradas=None, nClases=None):
    # Configuración ligeramente modificada para el reconocimiento de frases
    n_filtros_etapa = [16, 32, 64]  # Aumentamos filtros para capturar más información temporal
      
    entradas = layers.Input(shape=(formatoEntradas,), name='input_layer')  
    
    # Usamos nuestra capa personalizada en lugar de operaciones tf directas
    x = MelSpectrogramLayer(n_mel_bins=40)(entradas)
    
    print(f"Shape después de ExtractorRasgos: {tf.keras.backend.int_shape(x)}")
    
    x = layers.Dropout(0.2)(x)  # Aumentamos ligeramente el dropout

    for n_etapa in range(0, len(n_filtros_etapa)):
        n_filtros = n_filtros_etapa[n_etapa]
        
        x = layers.Conv2D(n_filtros, kernel_size=(3,3), strides=(1,1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Dropout(0.2)(x)
    
    print(f"Shape antes de Flatten: {tf.keras.backend.int_shape(x)}")
    
    # Usamos Flatten en lugar de Reshape
    x = layers.Flatten()(x)
    
    # Agregar una capa densa adicional para mejorar el reconocimiento
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Capa de salida para clasificar las frases
    clase = layers.Dense(nClases, activation='softmax')(x)

    modelo = Model(entradas, clase, name='cnn_frases')
 
    return modelo

def Modelo_CNN_Locutores(formatoEntradas=None, nClases=None):
    n_filtros_etapa= [4, 8, 16]
      
    entradas= layers.Input(shape=(formatoEntradas,), name='input_layer')  
    
    # Usamos nuestra capa personalizada en lugar de operaciones tf directas
    x = MelSpectrogramLayer(n_mel_bins=40)(entradas)


    # Añadir esta línea para depuración (opcional)
    print(f"Shape después de ExtractorRasgos: {tf.keras.backend.int_shape(x)}")
    
    x= layers.Dropout(0.1)(x)

    for n_etapa in range(0,len(n_filtros_etapa)):
        n_filtros= n_filtros_etapa[n_etapa]
        
        x= layers.Conv2D(n_filtros, kernel_size=(3,3), strides=(1,1), padding="same")(x)
        x= layers.BatchNormalization()(x)
        x= layers.Activation("relu")(x)
        x= layers.MaxPooling2D(pool_size=(2,2))(x)
        x= layers.Dropout(0.1)(x)
    
    # Añadir aquí para asegurarnos que las dimensiones son válidas
    print(f"Shape antes de Flatten: {tf.keras.backend.int_shape(x)}")
    
    # Asegurar que el tensor tiene una forma definida antes de aplanarlo
    x = layers.Reshape((-1,))(x)  # Reformar a un vector plano
    clase= layers.Dense(nClases, activation='softmax')(x)

    modelo= Model(entradas, clase, name='cnn')
 
    return modelo