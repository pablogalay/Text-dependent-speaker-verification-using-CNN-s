import numpy as np
import os
import sys
from EntrenamientoFrases import ModeloFrases
from EntrenamientoLocutor import ModeloLocutor
from TestFrases import TestFrases
from TestLocutor import TestLocutor

def entrenar_modelo_frases():
    """Entrenar el modelo de reconocimiento de frases"""
    print("\n=== ENTRENAMIENTO DEL MODELO DE FRASES ===\n")
    modelo_frases = ModeloFrases(ruta_audio='./train', epochs=20)
    modelo_frases.preparar_datasets()
    modelo_frases.cargar_o_crear_modelo()
    modelo_frases.entrenar_modelo()
    
    # Guardar la lista de frases para uso posterior en pruebas
    np.save('frases_lista.npy', modelo_frases.listaFrases_train_test)
    print("\nEntrenamiento del modelo de frases completado.")

def entrenar_modelo_locutores():
    """Entrenar el modelo de reconocimiento de locutores"""
    print("\n=== ENTRENAMIENTO DEL MODELO DE LOCUTORES ===\n")
    modelo_locutores = ModeloLocutor(ruta_audio='./train', epochs=20)
    modelo_locutores.preparar_datasets()
    modelo_locutores.cargar_o_crear_modelo()
    modelo_locutores.entrenar_modelo()
    
    # Guardar la lista de locutores para uso posterior en pruebas
    np.save('locutores_lista.npy', modelo_locutores.listaLocutores_train_test)
    print("\nEntrenamiento del modelo de locutores completado.")

def probar_modelo_frases(archivo_audio=None):
    """Probar el modelo de reconocimiento de locutores"""
    print("\n=== PRUEBA DEL MODELO DE LOCUTORES ===\n")
    
    try:
        locutores = np.load('locutores_lista.npy')
        print(f"Lista de locutores cargada.")
    except:
        locutores = None
        print("No se encontró el archivo de locutores, se utilizarán índices numéricos")
    
    test = TestLocutor(modelo_path='modelo_locutores_best.h5', locutores=locutores)

    test_folder = './test'
    total = 0
    
    for frase_dir in os.listdir(test_folder):
        ruta_frase = os.path.join(test_folder, frase_dir)
        if os.path.isdir(ruta_frase):
            for locutor_dir in os.listdir(ruta_frase):
                ruta_locutor = os.path.join(ruta_frase, locutor_dir)
                if os.path.isdir(ruta_locutor):
                    for file in os.listdir(ruta_locutor):
                        if file.lower().endswith('.wav'):
                            ruta_audio = os.path.join(ruta_locutor, file)
                            total += 1
                            resultado = test.reconocer_locutor_silencioso(ruta_audio)
                            locutor_real = locutor_dir
                            locutor_predicho = resultado['locutor'] if resultado else None

                            print(f"Audio: {file} -> Predicho: {locutor_predicho} | Real: {locutor_real}")

    print(f"\nTotal de audios analizados: {total}")

def probar_modelo_locutores():
    """Probar el modelo de reconocimiento de locutores"""
    print("\n=== PRUEBA DEL MODELO DE LOCUTORES ===\n")
    
    # Cargar la lista de locutores
    try:
        locutores = np.load('locutores_lista.npy')
        print(f"Lista de locutores cargada: {locutores}")
    except:
        locutores = None
        print("No se encontró el archivo de locutores, se utilizarán índices numéricos")
    
    # Crear el objeto de prueba
    test = TestLocutor(modelo_path='modelo_locutores_best.h5', locutores=locutores)

    test_folder = './test'  # Carpeta de test
    aciertos = 0
    total = 0
    
    for file in os.listdir(test_folder):
        ruta = os.path.join(test_folder, file)
        if os.path.exists(ruta):
            total += 1
            resultado = test.reconocer_locutor_silencioso(ruta)
            if resultado:
                print(f"Archivo: {file} -> Locutor reconocido: {resultado['locutor']}")
                # Aquí deberías comparar con la etiqueta real para saber si es acierto o no
                # Ejemplo ficticio (deberías definir locutor_real para cada archivo)
                # if resultado['locutor'] == locutor_real:
                #     aciertos += 1
            else:
                print(f"Archivo: {file} -> No se pudo reconocer el locutor.")

    print(f"\nArchivos analizados: {total}")

def probar_modelo_completo():
    """Probar el modelo de locutores y frases juntos"""
    print("\n=== PRUEBA DEL MODELO COMPLETO (LOCUTOR Y FRASE) ===\n")
    
    # Cargar los modelos una vez
    test_frases = TestFrases(modelo_path='modelo_frases_best.h5', frases=np.load('frases_lista.npy'))
    test_locutores = TestLocutor(modelo_path='modelo_locutores_best.h5', locutores=np.load('locutores_lista.npy'))

    test_folder = './test'
    total = 0
    aciertos_locutor = 0
    aciertos_frase = 0
    aciertos_ambos = 0

    for frase_dir in os.listdir(test_folder):  # Carpetas S1, S2, etc.
        ruta_frase = os.path.join(test_folder, frase_dir)
        if os.path.isdir(ruta_frase):
            for locutor_dir in os.listdir(ruta_frase):  # Carpetas spk_000001, etc.
                ruta_locutor = os.path.join(ruta_frase, locutor_dir)
                if os.path.isdir(ruta_locutor):
                    for file in os.listdir(ruta_locutor):  # Archivos .wav
                        if file.lower().endswith('.wav'):
                            ruta_audio = os.path.join(ruta_locutor, file)
                            total += 1
                            
                            frase_real = frase_dir
                            locutor_real = locutor_dir
                            
                            resultado_locutor = test_locutores.reconocer_locutor_silencioso(ruta_audio)
                            resultado_frase = test_frases.reconocer_frase_silencioso(ruta_audio)

                            locutor_correcto = False
                            frase_correcta = False

                            locutor_predicho = resultado_locutor['locutor'] if resultado_locutor else None
                            frase_predicha = resultado_frase['frase'] if resultado_frase else None

                            if locutor_predicho == locutor_real:
                                aciertos_locutor += 1
                                locutor_correcto = True

                            if frase_predicha == frase_real:
                                aciertos_frase += 1
                                frase_correcta = True

                            if locutor_correcto and frase_correcta:
                                aciertos_ambos += 1

                            #print(f"Audio: {ruta_audio}")
                            #print(f"  Locutor - Predicho: {locutor_predicho} | Real: {locutor_real}")
                            #print(f"  Frase   - Predicha: {frase_predicha} | Real: {frase_real}")
                            #print()

    porcentaje_aciertos_locutor = (aciertos_locutor / total) * 100 if total > 0 else 0
    porcentaje_aciertos_frase = (aciertos_frase / total) * 100 if total > 0 else 0
    porcentaje_aciertos_completo = (aciertos_ambos / total) * 100 if total > 0 else 0

    print("\n=== RESULTADOS ===")
    print(f"Total de muestras procesadas: {total}")
    print(f"Aciertos en locutor: {aciertos_locutor} ({porcentaje_aciertos_locutor:.2f}%)")
    print(f"Aciertos en frase: {aciertos_frase} ({porcentaje_aciertos_frase:.2f}%)")
    print(f"Aciertos en locutor y frase a la vez: {aciertos_ambos} ({porcentaje_aciertos_completo:.2f}%)")

def reconocer_locutor_aleatorio(test_folder='./test'):
    """Reconocer un locutor de un audio aleatorio mostrando todas las probabilidades"""
    import random
    
    # Crear el objeto de prueba
    try:
        locutores = np.load('locutores_lista.npy')
    except:
        locutores = None
        print("No se encontró el archivo de locutores, se utilizarán índices numéricos")
    
    test_locutores = TestLocutor(modelo_path='modelo_locutores_best.h5', locutores=locutores)

    rutas = []
    for frase_dir in os.listdir(test_folder):
        ruta_frase = os.path.join(test_folder, frase_dir)
        if os.path.isdir(ruta_frase):
            for locutor_dir in os.listdir(ruta_frase):
                ruta_locutor = os.path.join(ruta_frase, locutor_dir)
                if os.path.isdir(ruta_locutor):
                    for file in os.listdir(ruta_locutor):
                        if file.lower().endswith('.wav'):
                            rutas.append(os.path.join(ruta_locutor, file))
    
    if rutas:
        ruta_audio = random.choice(rutas)
        resultado = test_locutores.reconocer_locutor(ruta_audio)  # Usamos el objeto test_locutores
        print(f"\nAudio seleccionado: {ruta_audio}")
        
        if resultado is not None:
            #print("Probabilidades de cada locutor:")
            #print(resultado['probabilidades'])  # Aquí mostramos todas las probabilidades
            #print(f"Locutor más probable: {resultado['locutor']}")
            
            # Obtener el locutor real (nombre de la carpeta) para comparar
            locutor_real = ruta_audio.split("\\")[-2]  # Asumiendo que la estructura es ./test/[frase]/[locutor]/audio.wav
            print(f"Locutor real: {locutor_real}")
        else:
            print("No se pudo reconocer el locutor.")
    else:
        print("No se encontraron audios en la carpeta de test.")

def reconocer_frase_aleatoria(test_folder='./test'):
    """Reconocer una frase de un audio aleatorio mostrando todas las probabilidades"""
    import random
    
    # Crear el objeto de prueba
    try:
        frases = np.load('frases_lista.npy')
    except:
        frases = None
        print("No se encontró el archivo de frases, se utilizarán índices numéricos")
    
    test_frases = TestFrases(modelo_path='modelo_frases_best.h5', frases=frases)

    rutas = []
    for frase_dir in os.listdir(test_folder):
        ruta_frase = os.path.join(test_folder, frase_dir)
        if os.path.isdir(ruta_frase):
            for locutor_dir in os.listdir(ruta_frase):
                ruta_locutor = os.path.join(ruta_frase, locutor_dir)
                if os.path.isdir(ruta_locutor):
                    for file in os.listdir(ruta_locutor):
                        if file.lower().endswith('.wav'):
                            rutas.append(os.path.join(ruta_locutor, file))
    
    if rutas:
        ruta_audio = random.choice(rutas)
        resultado = test_frases.reconocer_frase(ruta_audio)  # Use the test_frases object
        print(f"\nAudio seleccionado: {ruta_audio}")
        
        if resultado is not None:
            #print("Probabilidades de cada frase:")
            #print(resultado['probabilidades'])  # Aquí mostramos todas las probabilidades
            #print(f"Frase más probable: {resultado['frase']}")
            
            # Obtener la frase real (nombre de la carpeta) para comparar
            frase_real = ruta_audio.split("\\")[-3]  # Asumiendo que la estructura es ./test/[frase]/[locutor]/audio.wav
            print(f"Frase real: {frase_real}")
        else:
            print("No se pudo reconocer la frase.")
    else:
        print("No se encontraron audios en la carpeta de test.")

def mostrar_menu():
    """Mostrar menú principal"""
    print("\n===== SISTEMA DE RECONOCIMIENTO DE VOZ =====")
    print("1. Entrenar modelo de frases")
    print("2. Entrenar modelo de locutores")
    print("3. Probar modelo de frases")
    print("4. Probar modelo de locutores")
    print("5. Probar modelo completo (locutor y frase)")
    print("6. Reconocer frase aleatoria")
    print("7. Reconocer locutor aleatorio")
    print("0. Salir")
    return input("\nSeleccione una opción: ")

if __name__ == "__main__":
    # Verificar si se ha pasado algún argumento por línea de comandos
    if len(sys.argv) > 1:
        comando = sys.argv[1].lower()
        
        if comando == "train-frases":
            entrenar_modelo_frases()
        elif comando == "train-locutores":
            entrenar_modelo_locutores()
        elif comando == "test-frases":
            if len(sys.argv) >= 3:
                probar_modelo_frases(sys.argv[2])
            else:
                probar_modelo_frases()
        elif comando == "test-locutores":
            if len(sys.argv) >= 3:
                probar_modelo_locutores(sys.argv[2])
            else:
                probar_modelo_locutores()
        elif comando == "completo":
            if len(sys.argv) >= 3:
                probar_modelo_completo(sys.argv[2])
            else:
                print("Error: Debe especificar un archivo de audio para el reconocimiento completo")
        elif comando == "reconocer-frase-aleatoria":
            reconocer_frase_aleatoria()
        elif comando == "reconocer-locutor-aleatorio":
            reconocer_locutor_aleatorio()
        else:
            print("Comando no reconocido.")
            print("Uso: python main.py [train-frases|train-locutores|test-frases|test-locutores|completo] [archivo_audio.wav (opcional)]")
    else:
        # Modo interactivo con menú
        opcion = ""
        while opcion != "0":
            opcion = mostrar_menu()
            
            if opcion == "1":
                entrenar_modelo_frases()
            elif opcion == "2":
                entrenar_modelo_locutores()
            elif opcion == "3":
                probar_modelo_frases()
            elif opcion == "4":
                probar_modelo_locutores()
            elif opcion == "5":
                probar_modelo_completo()
            elif opcion == "6":
                reconocer_frase_aleatoria()
            elif opcion == "7":
                reconocer_locutor_aleatorio()
            elif opcion == "0":
                print("Saliendo del programa...")
            else:
                print("Opción no válida, intente nuevamente.")
