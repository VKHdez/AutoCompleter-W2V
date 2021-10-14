
## Imports
import PySimpleGUI as sg # GUI Import
# Concurrency import
# regex import

from pathlib import Path
import string

import numpy as np
import re
import tensorflow as tf # Word2Vec import
from tensorflow.keras import layers

from Word2Vec import Word2Vec


# Funcion de Vector de entrada
def to_one_hot(data_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_index] = 1
    return temp


if __name__ == '__main__':

    # GUI Layout elementos
    layout = [
        [sg.Multiline(size=(80,30),font=('Arial',13))],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]

    # PREPARACIÓN DE DATOS
    corpus_raw = Path('libros/Abe Shana - La última sirena.txt').read_text()
    raw_sentences = list(re.findall(r"[\w']+",corpus_raw.lower()))  # Carga y separa en el texto en oraciones

    sentences = []

    #for sentence in raw_sentences:
    #    sentences.append(sentence.split())  # separa las oraciones en Palabras

    data = []
    window_size = 2

    for word_index, word in enumerate(sentences):
        for nb_word in sentences[max(word_index - window_size, 0): min(word_index + window_size, len(sentences)) + 1]:
            if nb_word != word:
                data.append([word, nb_word])

    words_list = []

    for word in raw_sentences:
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
        #x_train.append(1)
        x_train.append(to_one_hot(wordDictionary2[data_word[0]], vocab_size))
        y_train.append(to_one_hot(wordDictionary2[data_word[1]], vocab_size))

    x_train = np.asarray(x_train, dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")

    # FIN DE LA PREPARACIÓN DE DATOS

    w2v = Word2Vec(vocab_size=vocab_size,optimizer='adam', epochs=10000)
    w2v.entrenamiento(x_train, y_train)

    sg.theme('TanBlue')
    window = sg.Window("Word2Vec Completion", layout)

    while True:
        event, values = window.read()
        if event == "Cancel" or event == sg.WIN_CLOSED:
            break

    window.close()