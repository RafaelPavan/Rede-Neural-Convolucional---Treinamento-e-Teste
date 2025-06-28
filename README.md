# Rede-Neural-Convolucional-Treinamento-e-Teste

Projeto de um Rede Neural Convolucional para identificar dígitos de 0 a 9 por meio de imagem.

## Como Executar o Treinamento

Para treinar o modelo, siga os passos abaixo:

### Pré-requisitos

- **Python 3**: Instale o Python 3 (a versão utilizada no desenvolvimento foi a **3.12**).

- **Bibliotecas**: Instale as seguintes bibliotecas usando `pip`:

  - `TensorFlow`
  - `Numpy`
  - `Scikit-learn`

  Você pode instalá-las com o comando:

  ```bash
  pip install tensorflow numpy scikit-learn
  ```

### Execução

Após configurar os pré-requisitos, basta rodar o arquivo `treinamento_rede.py` via sua IDE ou terminal.

---

## Como Executar o Teste

Para testar o modelo, siga estas instruções:

### Pré-requisitos

Os pré-requisitos são os mesmos do treinamento. Se você já instalou as bibliotecas, não precisa fazer isso novamente.

### Execução

1.  **Arquivo de Teste**: O teste utiliza o arquivo `Numero.bmp` como base. Este arquivo contém uma imagem de um número (de 0 a 9).
2.  **Rodar o Código**: Execute o arquivo `teste_previsao.py` via sua IDE ou terminal.
3.  **Interatividade**:
    - O código irá logar informações sobre a imagem que está sendo lida.
    - Você pode abrir o arquivo `Numero.bmp` em um software de desenho (como o Paint), editar o número e salvar. O script `teste_previsao.py` detectará a alteração e identificará o novo valor, mostrando a porcentagem de confiança.
    - A linha **49** do código contém uma função `break` comentada. Como utilizamos um `while True` para manter o loop de detecção em tempo real, você pode descomentar essa linha para sair do loop a qualquer momento.

---
