from utils import *

def createVocabulary(stop_words):
    vocabulary = {}
    i = 1
    words_set = set()
    for file_id in range(1,30001):
        fname = f'tsv/{file_id}.tsv'
        if os.path.isfile(fname):
            file1 = open(fname)
            line = file1.read()
            sorted_words = processWords(line, stop_words)
            words_set |= set(sorted_words)
    for word in words_set:
        vocabulary[word] = i
        i += 1
    with open('vocabulary.json', 'w') as fp:
        json.dump(vocabulary, fp)
        
    return vocabulary

def createInvertedIndex(vocabulary):
    inverted_index = defaultdict(set)
    i = 1
    for file_id in range(1,30001):
        fname = f'tsv/{file_id}.tsv'
        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                words = processWords(file.read(), stop_words)
            for w in words:
                term_id = vocabulary[w] 
                inverted_index[term_id].add(file_id)

    for i in inverted_index:
        inverted_index[i] = sorted(list(inverted_index[i]))

    with open('inverted_index.json', 'w') as fp:
        json.dump(inverted_index, fp)
        
    return inverted_index


def create_inverted_index_2(fields, vocabulary, norm_file=None, index_file=None):
    inverted_index_2 = defaultdict(list)
    i = 1
    doc_number = 30001
    doc_norm = defaultdict(list)

    for file_id in range(1, doc_number):
        fname = f'tsv/{file_id}.tsv'
        if os.path.isfile(fname):
            text = data_from_tsv(fname, fields)

            #  Extract usefull terms from text 
            words = processWords(text, stop_words)

            #  Convert text represantion of words into ids of vacabulary
            words = words_into_id(words, vocabulary)

            #  Create the frequency dicionary 'term: count'
            counter = Counter(words)

            for word_id, tf in counter.items():
                inverted_index_2[word_id].append([file_id, tf ])
                doc_norm[file_id].append((word_id, tf))


    #  Construct documets norms
    for file_id in doc_norm:
        norm = 0
        for word_id, tf in doc_norm[file_id]:
            norm += (tf * get_idf(word_id,inverted_index_2, doc_number)) ** 2
        doc_norm[file_id] = math.sqrt(norm)

    if norm_file:
        #  Store norms of documents as json
        with open(norm_file, 'w') as fp:
            json.dump(doc_norm, fp)

    #  Mutiplying each tf by corresponding idf
    for word_id in inverted_index_2:
        for i in range(len(inverted_index_2[word_id])):
            inverted_index_2[word_id][i][1] *= get_idf(word_id,inverted_index_2, len(doc_norm))
            inverted_index_2[word_id][i] = tuple(inverted_index_2[word_id][i])

    if index_file:
        #  Store inverted index as json
        with open(index_file, 'w') as fp:
            json.dump(inverted_index_2, fp)

    return inverted_index_2, doc_norm