import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import Model # type: ignore
from tensorflow.keras import layers # type: ignore
from PreprocesamientoDatos import decodificar_audio, procesar_audio_dataset, obtenerFicheros, get_frase
from ModeloNeuronal import Modelo_CNN_Frases, MelSpectrogramLayer
import matplotlib.pyplot as plt
from collections import Counter

class ModeloFrases:
    def __init__(self, 
                 ruta_audio='./train', 
                 patron_audio='*.wav',
                 batch_size_train=32,
                 batch_size_test=128,
                 learning_rate=0.0003,
                 epochs=20,
                 shuffle_buffer=65536,
                 semilla=99):
        
        # Constantes
        self.SEMILLA = semilla
        self.FRECUENCIA_MUESTREO = 16000
        self.DOS_SEGUNDOS = 2
        self.MUESTRAS_2S = (self.FRECUENCIA_MUESTREO * self.DOS_SEGUNDOS)
        self.SHUFFLE_BUFFER_SIZE = shuffle_buffer
        self.TRAIN_BATCH_SIZE = batch_size_train
        self.TEST_BATCH_SIZE = batch_size_test
        self.LR = learning_rate
        self.EPOCHS = epochs
        
        # Rutas y patrones
        self.ruta_audio = ruta_audio
        self.patron_audio = patron_audio
        
        # Configurar semillas para reproducibilidad
        tf.keras.utils.set_random_seed(self.SEMILLA)
        tf.random.set_seed(self.SEMILLA)
        
        # Inicializar listas y modelo
        self.listaFicheros_train_test = []
        self.listaFrasesFicheros_train_test = []
        self.listaFrases_train_test = None
        self.modelo = None
        
        # Cargar y preprocesar datos
        self._cargar_datos()
        
    def _cargar_datos(self):
        """Cargar archivos de audio y generar listas de frases"""
        # Recorremos los archivos y obtenemos la frase (S1, S2, etc.)
        for p in Path(self.ruta_audio).rglob(self.patron_audio):
            if p.is_file():
                self.listaFicheros_train_test.append(str(p))
                self.listaFrasesFicheros_train_test.append(p.parent.parent.name)
                
        self.n_listaFicheros_train_test = len(self.listaFicheros_train_test)    
        self.listaFrases_train_test = np.unique(self.listaFrasesFicheros_train_test)
        self.n_frases_train_test = len(self.listaFrases_train_test)
        
        print(f"Total de frases encontradas: {self.n_frases_train_test}")
        print(f"Frases: {self.listaFrases_train_test}")
        
        # Asignar los archivos de entrenamiento y prueba desde las rutas respectivas
        self.listaFicheros_train = [str(p) for p in Path('./train').rglob(self.patron_audio)]
        self.listaFrasesFicheros_train = [p.parent.parent.name for p in Path('./train').rglob(self.patron_audio)]
        self.listaClasesFicheros_train = [np.where(self.listaFrases_train_test == l)[0][0] for l in self.listaFrasesFicheros_train]
        
        self.listaFicheros_test = [str(p) for p in Path('./test').rglob(self.patron_audio)]
        self.listaFrasesFicheros_test = [p.parent.parent.name for p in Path('./test').rglob(self.patron_audio)]
        self.listaClasesFicheros_test = [np.where(self.listaFrases_train_test == l)[0][0] for l in self.listaFrasesFicheros_test]
        
        # Mostrar estadísticas de la división
        self._mostrar_estadisticas()
        
    def _mostrar_estadisticas(self):
        """Mostrar estadísticas de la división de datos"""
        train_phrases = Counter(self.listaFrasesFicheros_train)
        test_phrases = Counter(self.listaFrasesFicheros_test)
        
        print('\n')
        print('--------------------------------------------------------------------------')
        print(f'Train: {len(self.listaFicheros_train)} ficheros')
        print(f'Test: {len(self.listaFicheros_test)} ficheros')
        print('Distribución de frases en entrenamiento:')
        for frase, cantidad in train_phrases.items():
            print(f'  {frase}: {cantidad} ejemplos')
        print('Distribución de frases en prueba:')
        for frase, cantidad in test_phrases.items():
            print(f'  {frase}: {cantidad} ejemplos')
        print('--------------------------------------------------------------------------')
        print('\n')
        
    def preparar_datasets(self):
        """Preparar los conjuntos de datos para entrenamiento y evaluación"""
        # Crear opciones comunes para los datasets
        opciones = tf.data.Options()
        opciones.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        
        # Dataset de entrenamiento
        ObtenerFicherosTrain = obtenerFicheros(self.listaFicheros_train, self.listaClasesFicheros_train)
        self.train_dataset = tf.data.Dataset.from_generator(
            lambda: ObtenerFicherosTrain,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        # Orden: map -> cache -> shuffle -> repeat -> batch -> prefetch
        self.train_dataset = self.train_dataset.map(procesar_audio_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_dataset = self.train_dataset.cache()
        self.train_dataset = self.train_dataset.shuffle(self.SHUFFLE_BUFFER_SIZE, self.SEMILLA)
        self.train_dataset = self.train_dataset.repeat()
        self.train_dataset = self.train_dataset.batch(self.TRAIN_BATCH_SIZE)
        self.train_dataset = self.train_dataset.prefetch(tf.data.AUTOTUNE)
        self.train_dataset = self.train_dataset.with_options(opciones)
        
        # Dataset para evaluación completa
        self.eval_dataset = tf.data.Dataset.from_generator(
            lambda: obtenerFicheros(self.listaFicheros_test, self.listaClasesFicheros_test),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        self.eval_dataset = self.eval_dataset.map(procesar_audio_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        self.eval_dataset = self.eval_dataset.batch(self.TEST_BATCH_SIZE)
        self.eval_dataset = self.eval_dataset.cache().prefetch(tf.data.AUTOTUNE)
        self.eval_dataset = self.eval_dataset.with_options(opciones)
        
        # Dataset para validación durante entrenamiento (con límite y repetición)
        limite_test = min(1000, len(self.listaFicheros_test))
        self.test_dataset = tf.data.Dataset.from_generator(
            lambda: obtenerFicheros(self.listaFicheros_test[:limite_test], 
                                    self.listaClasesFicheros_test[:limite_test]),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        self.test_dataset = self.test_dataset.map(procesar_audio_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.cache()
        self.test_dataset = self.test_dataset.batch(self.TEST_BATCH_SIZE)
        self.test_dataset = self.test_dataset.repeat(-1)
        self.test_dataset = self.test_dataset.prefetch(tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.with_options(opciones)
        
    def cargar_o_crear_modelo(self, modelo_path='modelo_frases_best.h5'):
        """Cargar un modelo existente o crear uno nuevo"""
        estrategia = tf.distribute.OneDeviceStrategy(device="/device:GPU:0")
        with estrategia.scope():
            print('Red neuronal para reconocimiento de frases [TensorFlow %s]' % (tf.__version__))
            print('Numero de GPUs: {}'.format(estrategia.num_replicas_in_sync))
            
            if os.path.exists(modelo_path):
                print(f"Cargando modelo previo desde: {modelo_path}")
                try:
                    # Registrar las capas personalizadas para la carga del modelo
                    custom_objects = {
                        'MelSpectrogramLayer': MelSpectrogramLayer,
                        # Para compatibilidad con modelos antiguos
                        'ExtractorRasgos': MelSpectrogramLayer
                    }
                    
                    # Intentar cargar el modelo con las capas personalizadas
                    with tf.keras.utils.custom_object_scope(custom_objects):
                        self.modelo = tf.keras.models.load_model(modelo_path)
                    print("Modelo cargado exitosamente.")
                except Exception as e:
                    print(f"Error al cargar modelo: {e}")
                    print("Creando un nuevo modelo...")
                    self.modelo = Modelo_CNN_Frases(self.MUESTRAS_2S, self.n_frases_train_test)
            else:
                print("No se encontró un modelo previo. Creando un modelo nuevo.")
                self.modelo = Modelo_CNN_Frases(self.MUESTRAS_2S, self.n_frases_train_test)
            
            # Mostrar resumen del modelo
            self.modelo.summary()
            
            # Compilar el modelo
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.LR)
            self.modelo.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
    def entrenar_modelo(self, modelo_path='modelo_frases_best.h5'):
        """Entrenar el modelo o continuar entrenamiento"""
        # Configurar callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=modelo_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
        
        # Configuración de pasos por época
        pasos_por_epoca = max(1, len(self.listaFicheros_train) // self.TRAIN_BATCH_SIZE)
        validation_steps = max(1, len(self.listaFicheros_test) // self.TEST_BATCH_SIZE)
        
        # Preguntar al usuario si desea continuar con el entrenamiento
        continuar_entrenamiento = True
        if os.path.exists(modelo_path) and self.modelo is not None:
            respuesta = input("Se ha cargado un modelo previo. ¿Desea continuar con el entrenamiento? (s/n): ").lower()
            continuar_entrenamiento = respuesta in ['s', 'si', 'sí', 'y', 'yes', '']
        
        if continuar_entrenamiento:
            print("\nIniciando entrenamiento...")
            try:
                # Entrenar el modelo
                self.history = self.modelo.fit(
                    self.train_dataset, 
                    epochs=self.EPOCHS, 
                    validation_data=self.test_dataset,
                    steps_per_epoch=pasos_por_epoca,
                    validation_steps=validation_steps,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluar el modelo
                print("\nEvaluando el modelo con el conjunto de prueba...")
                test_loss, test_acc = self.modelo.evaluate(self.eval_dataset, verbose=2)
                print('\nAccuracy de reconocimiento de frases:', test_acc)
                
                # Guardar el modelo entrenado con un nombre diferente
                self.modelo.save('modelo_reconocedor_frases_final.h5')
                print("Modelo guardado como 'modelo_reconocedor_frases_final.h5'")
                
                # Visualizar resultados
                self.visualizar_resultados()
                
            except KeyboardInterrupt:
                print("\nEntrenamiento interrumpido manualmente.")
            except Exception as e:
                print(f"\nError durante el entrenamiento: {e}")
        else:
            print("\nEntrenamiento cancelado. Usando el modelo previamente cargado.")
            # Evaluar el modelo sin entrenarlo más
            print("Evaluando el modelo con el conjunto de prueba...")
            test_loss, test_acc = self.modelo.evaluate(self.eval_dataset, verbose=2)
            print('\nAccuracy de reconocimiento de frases:', test_acc)
    
    def visualizar_resultados(self):
        """Visualizar curvas de aprendizaje del modelo"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Precisión del modelo')
        plt.ylabel('Precisión')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Pérdida del modelo')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('resultados_reconocedor_frases.png')
        plt.show()