from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import numpy as np
import os
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

num_classes = 10 # De 0 a 9

PATH = os.getcwd()
caminho_treinamento = os.path.join(PATH, 'data')
diretorio_lista = os.listdir(caminho_treinamento)

def buscar_diretorios():
    diretorios_validos = []

    for d in diretorio_lista:
        caminho_completo = os.path.join(caminho_treinamento, d)

        if os.path.isdir(caminho_completo):
            diretorios_validos.append(d)
    
    return sorted(diretorios_validos)

def carregamento_e_pre_processar_dados(caminho_treinamento, num_classes):
    lista_dados_de_imagens = []
    lista_de_rotulos = []
    
    lista_dados_de_imagens_ordenada = buscar_diretorios()

    for i, dataset in enumerate(lista_dados_de_imagens_ordenada):
        caminho_classe = os.path.join(caminho_treinamento, dataset)
        img_list = os.listdir(caminho_classe)
        print(f'Carregando imagens do dataset: {dataset}')
        for nome_img in img_list:
            caminho_img = os.path.join(caminho_classe, nome_img)
            input_img = cv2.imread(caminho_img)
            if input_img is None:
                print(f"Não foi possível carregar a imagem {caminho_img}.")
                continue

            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_ajustada = cv2.resize(input_img, (128, 128))
            lista_dados_de_imagens.append(input_img_ajustada)
            lista_de_rotulos.append(i)

    dados_img = np.array(lista_dados_de_imagens)
    rotulos = np.array(lista_de_rotulos)

    dados_img = dados_img.astype('float32') / 255.0

    dados_img = np.expand_dims(dados_img, axis=-1)

    Y = to_categorical(rotulos, num_classes)

    x, y = shuffle(dados_img, Y, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    input_forma = X_train.shape[1:] 
    return X_train, X_test, y_train, y_test, input_forma

def definir_modelo_cnn(input_forma, num_classes):
    modelo = keras.Sequential([
    keras.Input(shape=input_forma),

    layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='detecta_bordas_e_texturas'),
    layers.MaxPooling2D(pool_size=(2, 2), name='reduz_mapa_caracteristicas_1'),

    layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='refina_padroes_locais'),
    layers.MaxPooling2D(pool_size=(2, 2), name='reduz_mapa_caracteristicas_2'),

    layers.Conv2D(16, (3, 3), activation='relu', name='captura_detalhes_finais'),
    layers.MaxPooling2D(pool_size=(2, 2), name='reduz_mapa_caracteristicas_3'),

    layers.Dropout(0.2, name='evita_sobreajuste_inicial'), 

    layers.Flatten(name='prepara_para_decisao_final'),
    
    layers.Dense(128, activation='relu', name='primeira_camada_decisao'),

    layers.Dense(64, activation='relu', name='segunda_camada_decisao'),

    layers.Dense(32, activation='relu', name='terceira_camada_decisao'),

    layers.Dropout(0.5, name='evita_sobreajuste_final'), 
    
    layers.Dense(num_classes, activation='softmax', name='saida_probabilidades_classes')
])

    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    modelo.summary()
    return modelo

def treinar_modelo(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train,
               batch_size=16,
               epochs=200,
               verbose=1,
               validation_data=(X_test, y_test))

    caminho_modelo_salvo = 'modelo_digitos.keras'
    modelo.save(caminho_modelo_salvo)

if __name__ == "__main__":
 
    X_train, X_test, y_train, y_test, input_forma = carregamento_e_pre_processar_dados(
        caminho_treinamento, num_classes
    )

    modelo = definir_modelo_cnn(input_forma, num_classes)

    treinar_modelo(modelo, X_train, y_train, X_test, y_test)