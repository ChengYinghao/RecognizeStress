import codecs
import numpy as np

from NeuralNetwork.Datas_first import convert_data

lexicon = np.array(['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р',
                    'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я'])
lexicon = lexicon.reshape((33, 1))


def read_file(path):
    words_file = []
    accents_file = []
    with codecs.open(path, 'r', 'utf-8-sig') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            word_tupe = line.split()
            words_file.append(word_tupe[0])
            accents_file.append(word_tupe[1])
    return words_file, accents_file


def load_datas():
    words_file, accents_file = read_file("semiData.txt")
    m = len(words_file)
    X = np.zeros((21, m), dtype=np.float32)
    Y = np.zeros((33, m), dtype=np.float32)
    for i in range(m):
        word = words_file[i]
        accent = accents_file[i]
        X[:, i], Y[:, i] = convert_data(word, accent)
    return X, Y
