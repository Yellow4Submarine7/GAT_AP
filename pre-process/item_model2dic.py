import gensim
from gensim.models import Word2Vec
from sklearn import model_selection
from gensim.models import Word2Vec
import gensim

model = Word2Vec.load("iterm2vec.model").wv

item_dic = gensim.models.KeyedVectors.load_word2vec_format("iterm2vec.model", binary=False)