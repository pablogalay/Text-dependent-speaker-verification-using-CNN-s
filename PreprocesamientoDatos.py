import tensorflow as tf
import numpy as np
import os
 
from pathlib import Path
from tensorflow.keras import Model # type: ignore
from tensorflow.keras import layers # type: ignore

TIEMPO_MUESTRA = 2 
FRECUENCIA_MUESTREO = 16000 
MUESTRAS_2S = TIEMPO_MUESTRA * FRECUENCIA_MUESTREO
        
def get_locutor(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    locutor = parts[-2]
    return locutor

def get_frase(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    frase = parts[-3]
    return frase

def decodificar_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_waveform_and_labels(file_path):
    frase = get_frase(file_path)
    locutor = get_locutor(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decodificar_audio(audio_binary)
    # Asegurar que waveform es float32
    waveform = tf.cast(waveform, tf.float32)
    return waveform, frase, locutor

def adjust_waveform_length(waveform):
    input_len = MUESTRAS_2S
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    return equal_length

def get_spectrogram(waveform):
    waveform = adjust_waveform_length(waveform)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_and_label_id(waveform, label, commands):
    spectrogram = get_spectrogram(waveform)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def obtenerFicheros(listaFicheros=None, listaClases=None):  
    for x in range(0, len(listaFicheros)):
        yield listaFicheros[x], listaClases[x]

def procesar_audio_dataset(file_path, label):
    # Leer el archivo de audio
    audio_binary = tf.io.read_file(file_path)
    # Decodificar el audio
    waveform = decodificar_audio(audio_binary)
    # Ajustar la longitud
    waveform = adjust_waveform_length(waveform)
    # Devolver la forma de onda ajustada y la etiqueta
    return waveform, label

def get_mel_spectrogram(waveform, n_mel_bins=40):
    """
    Convierte una forma de onda en un espectrograma de mel.
    """
    waveform = adjust_waveform_length(waveform)
    
    # Calcular STFT con parámetros específicos para CNN

    spectrogram = tf.signal.stft(
        waveform, 
        frame_length=400,
        frame_step=160,
        fft_length=512
    )
    
    # Obtener espectrograma de magnitud
    spectrogram = tf.abs(spectrogram)
    
    # Crear matriz de conversión a escala Mel
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel_bins, # Numero de filtros mel
        num_spectrogram_bins=257,  # Número de bins en el espectrograma
        sample_rate=FRECUENCIA_MUESTREO, # Frecuencia de muestreo
        lower_edge_hertz=0, # Frecuencia mínima
        upper_edge_hertz=FRECUENCIA_MUESTREO / 2 # Frecuencia máxima
    )
    
    # Convertir a escala mel
    mel_spectrogram = tf.matmul(spectrogram, mel_matrix)
    
    # Convertir a escala logarítmica
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # Normalizar
    normalized = (log_mel_spectrogram - tf.reduce_mean(log_mel_spectrogram)) / (tf.math.reduce_std(log_mel_spectrogram) + 1e-6)
    
    # Añadir dimensión para canales
    normalized = normalized[..., tf.newaxis]
    
    return normalized