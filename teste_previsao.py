from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import time

print("\n--- Teste de Previsão de Imagem Única ---")
caminho_numero_a_ser_descoberto = 'Numero.bmp'

def verificacao_e_tratamento_img(caminho_img, altura, largura):
    if not os.path.exists(caminho_img):
        print(f"Arquivo '{caminho_img}' não encontrado.")
        return None

    img_crua = cv2.imread(caminho_img)
    if img_crua is None:
        print(f"Não foi possível carregar a imagem '{caminho_img}'. Verifique o caminho ou o formato.")
        return None

    teste_img_cinza = cv2.cvtColor(img_crua, cv2.COLOR_BGR2GRAY)
    teste_img_ajustada = cv2.resize(teste_img_cinza, (largura, altura))
    teste_img_normal = teste_img_ajustada.astype('float32') / 255.0
    
    return np.expand_dims(np.expand_dims(teste_img_normal, axis=-1), axis=0)
     
def verificar_previsao():

    while True:

        modelo = load_model('modelo_digitos.keras')
        teste_img_processada = verificacao_e_tratamento_img(
            caminho_numero_a_ser_descoberto, 128, 128
        )

        if teste_img_processada is None:
            break 

        percentuais = modelo.predict(teste_img_processada)
        posicao = np.argmax(percentuais, axis=1)[0]
        
        print(f'\nPrevisão para {caminho_numero_a_ser_descoberto}:')
        if percentuais[0][posicao] > 0.50:
            print(f'Número: {posicao} com {percentuais[0][posicao]*100:.2f}% de confiança.')
        else:
            print(f'Desconhecido. Maior probabilidade para {posicao} com {percentuais[0][posicao]*100:.2f}%. Melhore sua caligrafia!')

        time.sleep(5)
        # Para sair do loop após a primeira previsão descomentar a linha abaixo
        # break

if __name__ == "__main__":
    verificar_previsao()
