import numpy as np

from NeuralNetwork.Datas_first import convert_data

lexicon = np.array(['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р',
                    'с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я'])
word = 'достопримечательность'
a = 'карьера 3'
b = np.zeros((21, 2))
c = np.zeros((33, 2))
d, v = convert_data('карьера', 3)
word = 'подушка'
length = len(word)
np.array([a for letter in word for a in range(len(lexicon)) if lexicon[a]==letter]).reshape((length, 1))
g = np.array([1 ,2 ,3 ,4, 5])
print(g[1:2])