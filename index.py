from index_utils import *


vocabulary = createVocabulary(stop_words)
print('Vocabulary is created')

#  Create index without score
createInvertedIndex(vocabulary)
print('Index for search 1 is created')

#  Create index and norm for intro + plot names
create_inverted_index_2([1,2], vocabulary, norm_file='document_norm.json', index_file='inverted_index_2.json')
print('Index for search 2 is created')

#  Create index and norm for file names
create_inverted_index_2([3], vocabulary, norm_file='name_norm.json', index_file='name_index.json')
print('Index for search 3 is created')
