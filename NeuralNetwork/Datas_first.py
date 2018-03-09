import os
import codecs
import numpy as np

lexicon = np.array(['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р',
           'с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я'])
lexicon = lexicon.reshape((33, 1))

accent_help = np.array([i - 1 for i in range(1, 34)])
def read_paths(rootDir):
    paths = []
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


# paths = read_paths("C:\学习\俄语单词")
# print(paths)


def read_file(path):
    words_file = []
    accents_file = []
    with codecs.open(path, 'r', 'UTF-8') as file_to_read:
        indice = 1
        while True:
            line = file_to_read.readline()
            if not line:
                break
            if (indice == 1):
                words_file.append(line)
                indice += 1
            elif (indice == 2):
                accents_file.append(line.split()[0])
                indice += 1
            elif (indice == 3):
                indice = 1
    return words_file, accents_file


def read_datas(rootDir):
    paths = read_paths(rootDir)
    words = []
    accents = []
    for path in paths:
        words_file, accents_file = read_file(path)
        words.extend(words_file)
        accents.extend(accents_file)
    return words, accents

# 该函数用来处理单个数据，将其转换为可以输入的向量
def convert_data(word, accent):
    accent = int(accent)
    indices_word = np.zeros((21, 1))
    length = len(word)
    indices_word[0:length] = np.array([a for letter in word for a in range(len(lexicon)) if lexicon[a]==letter.lower()],dtype=np.float32).reshape((length, 1))
    indice_accent = np.array((indices_word[accent] == accent_help)*1, dtype=np.float32)
    indice_accent = indice_accent.reshape((33, ))
    return indices_word.reshape((21,)), indice_accent