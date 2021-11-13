
import numpy as np
import re

from Word2Vec import Word2Vec

def to_one_hot(data_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_index] = 1
    return temp


if __name__ == '__main__':

    corpus_raw = 'Hola que tal, bienvenidos, esto es un test de string para probar el entrenamiento de nuestra pequeña implementación de Word2Vec'

    words = list(re.findall(r"[\w']+", corpus_raw.lower()))  # Carga y separa en el texto en oraciones

    data = []
    window_size = 2

    for word_index, word in enumerate(words):
        for nb_word in words[max(word_index - window_size, 0): min(word_index + window_size, len(words)) + 1]:
            if nb_word != word:
                data.append([word, nb_word])

    words_list = []

    for word in words:
        words_list.append(word)

    words_list = set(words_list)

    wordDictionary1 = {}
    wordDictionary2 = {}

    vocab_size = len(words_list)
    for index, word in enumerate(words_list):
        wordDictionary1[word] = index
        wordDictionary2[index] = word

    x_train = []
    y_train = []

    for data_word in data:
        x_train.append(to_one_hot(wordDictionary1[data_word[0]], vocab_size))
        y_train.append(to_one_hot(wordDictionary1[data_word[1]], vocab_size))

    x_train = np.asarray(x_train, dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")

    # FIN DE LA PREPARACIÓN DE DATOS

    # ENTRENAMIENTO

    w2v = Word2Vec(vocab_size=vocab_size, optimizer='adam', epochs=10000)
    w2v.entrenamiento(x_train, y_train)

    print("\n\n\n\tTerminada la sección de entrenamiento\n\n\n")

    print( w2v.vectorizacion(wordDictionary1['esto']) )
