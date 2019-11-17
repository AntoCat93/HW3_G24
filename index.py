from utils import *
from index_utils import *

createVocabulary(stop_words)

#  Create index without score
createInvertedIndex(vocab)

#  Create index and norm for intro + plot names
create_inverted_index_2([1,2], 
                       norm_file='document_norm.json', index_file='inverted_index_2.json')

#  Create index and norm for file names
create_inverted_index_2([3], vocabulary, norm_file='name_norm.json', index_file='name_index.json')

