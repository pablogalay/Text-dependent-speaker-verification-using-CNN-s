import tensorflow as tf
import numpy as np
import os
from PreprocesamientoDatos import decodificar_audio
from ModeloNeuronal import MelSpectrogramLayer

class TestFrases:
    def __init__(self, modelo_path='modelo_frases_best.h5', frases=None):
        self.FRECUENCIA_MUESTREO = 16000
        self.DOS_SEGUNDOS = 2
        self.MUESTRAS_2S = (self.FRECUENCIA_MUESTREO * self.DOS_SEGUNDOS)
        
        # Cargar lista de frases si se proporciona
        self.listaFrases_train_test = frases
        
        # Cargar el modelo
        self.cargar_modelo(modelo_path)
        
    def cargar_modelo(self, modelo_path):
        """Cargar un modelo preentrenado para pruebas"""
        if os.path.exists(modelo_path):
            print(f"Cargando modelo desde: {modelo_path}")
            try:
                # Registrar las capas personalizadas para la carga del modelo
                custom_objects = {
                    'MelSpectrogramLayer': MelSpectrogramLayer,
                    'ExtractorRasgos': MelSpectrogramLayer
                }
                
                # Cargar el modelo con las capas personalizadas
                with tf.keras.utils.custom_object_scope(custom_objects):
                    self.modelo = tf.keras.models.load_model(modelo_path)
                print("Modelo cargado exitosamente.")
                return True
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
                return False
        else:
            print(f"No se encontró el modelo en {modelo_path}")
            return False
    
    def reconocer_frase(self, ruta_audio):
        """Reconocer la frase hablada en un archivo de audio"""
        # Verificar que el modelo está cargado
        if not hasattr(self, 'modelo'):
            print("Error: No hay un modelo cargado")
            return None
            
        # Verificar que la lista de frases está disponible
        if self.listaFrases_train_test is None:
            print("Advertencia: No hay una lista de frases disponible")
            
        # Cargar y preprocesar el audio
        audio_binary = tf.io.read_file(ruta_audio)
        waveform = decodificar_audio(audio_binary)
        
        # Ajustar la duración
        n_muestras = tf.shape(waveform)[0]
        if n_muestras >= self.MUESTRAS_2S:
            waveform = waveform[:self.MUESTRAS_2S]
        else:
            zero_padding = tf.zeros(self.MUESTRAS_2S - n_muestras, tf.float32)
            waveform = tf.concat([waveform, zero_padding], axis=0)
        
        # Convertir a batch de 1
        waveform = tf.expand_dims(waveform, 0)
        
        # Predecir
        prediccion = self.modelo.predict(waveform)
        
        # Obtener la frase más probable
        frase_id = np.argmax(prediccion[0])
        probabilidad = prediccion[0][frase_id]
        
        # Mostrar los resultados
        resultado = {
            'frase_id': int(frase_id),
            'probabilidad': float(probabilidad),
            'todas_probabilidades': prediccion[0].tolist()
        }
        
        # Agregar el nombre de la frase si está disponible
        if self.listaFrases_train_test is not None:
            resultado['frase'] = self.listaFrases_train_test[frase_id]
            print(f"Frase reconocida: {self.listaFrases_train_test[frase_id]}")
            print(f"Probabilidad: {probabilidad*100:.2f}%")
            
            # Mostrar todas las probabilidades
            print("\nProbabilidades para todas las frases:")
            for i, frase in enumerate(self.listaFrases_train_test):
                print(f"{frase}: {prediccion[0][i]*100:.2f}%")
        else:
            print(f"Clase predicha: {frase_id}")
            print(f"Probabilidad: {probabilidad*100:.2f}%")
        
        return resultado
    
    def reconocer_frase_silencioso(self, ruta_audio):
        """Reconocer la frase hablada en un archivo de audio (versión silenciosa)"""
        # Verificar que el modelo está cargado
        if not hasattr(self, 'modelo'):
            print("Error: No hay un modelo cargado")
            return None
            
        # Verificar que la lista de frases está disponible
        if self.listaFrases_train_test is None:
            print("Advertencia: No hay una lista de frases disponible")
            
        # Cargar y preprocesar el audio
        audio_binary = tf.io.read_file(ruta_audio)
        waveform = decodificar_audio(audio_binary)
        
        # Ajustar la duración
        n_muestras = tf.shape(waveform)[0]
        if n_muestras >= self.MUESTRAS_2S:
            waveform = waveform[:self.MUESTRAS_2S]
        else:
            zero_padding = tf.zeros(self.MUESTRAS_2S - n_muestras, tf.float32)
            waveform = tf.concat([waveform, zero_padding], axis=0)
        
        # Convertir a batch de 1
        waveform = tf.expand_dims(waveform, 0)
        
        # Predecir
        prediccion = self.modelo.predict(waveform, verbose=0)  # <--- Silenciar predict también
        
        # Obtener la frase más probable
        frase_id = np.argmax(prediccion[0])
        probabilidad = prediccion[0][frase_id]
        
        # Preparar resultados
        resultado = {
            'frase_id': int(frase_id),
            'probabilidad': float(probabilidad),
            'todas_probabilidades': prediccion[0].tolist()
        }
        
        # Agregar el nombre de la frase si está disponible
        if self.listaFrases_train_test is not None:
            resultado['frase'] = self.listaFrases_train_test[frase_id]
        
        return resultado
        
    
    def evaluar_batch(self, lista_archivos, lista_clases=None):
        """Evaluar un conjunto de archivos de audio"""
        resultados = []
        aciertos = 0
        total = len(lista_archivos)
        
        for i, archivo in enumerate(lista_archivos):
            print(f"\nProcesando archivo {i+1}/{total}: {archivo}")
            resultado = self.reconocer_frase(archivo)
            
            # Verificar si la predicción es correcta (si se proporcionan clases)
            if lista_clases is not None and i < len(lista_clases):
                clase_real = lista_clases[i]
                es_correcto = resultado['frase_id'] == clase_real
                resultado['clase_real'] = int(clase_real)
                resultado['es_correcto'] = es_correcto
                
                if es_correcto:
                    aciertos += 1
                    print("✓ Predicción correcta")
                else:
                    print(f"✗ Predicción incorrecta. Clase real: {clase_real}")
            
            resultados.append(resultado)
        
        # Calcular precisión si se proporcionaron clases
        if lista_clases is not None:
            precision = aciertos / total
            print(f"\nPrecisión del modelo: {precision*100:.2f}% ({aciertos}/{total})")
        
        return resultados