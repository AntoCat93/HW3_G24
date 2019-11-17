from utils import *

with open('df_movies.json', 'r') as fp:
    df_movies = pd.read_json(json.load(fp))


vocabulary = load_vocabulary()


inverted_index = load_inverted_index_1()

inverted_index_2 = load_inverted_index_2()
doc_norm = load_doc_norm()
        
name_index = load_from_json('name_index.json')        
name_norm = load_from_json('name_norm.json')


def search_1(query, top = 10):
    return top_k(search_no_score(query, vocabulary, stop_words, inverted_index), df_movies, top, no_score=True)

def search_2(query, top = 10):
    return top_k(search_with_tfidf(query, vocabulary, inverted_index_2, doc_norm), df_movies, top)

def search_3(query, top = 10):
    #  as_heap = False ==> returns scores as dictionary {document_id : score}
    #  strict_terms_filter = False ==> tells function to use not_strict_filter_documents
    name_results = search_with_tfidf(query, vocabulary, name_index, doc_norm, as_heap=False, strict_terms_filter=False)
    #  additional_score will be added to scores with the same document id, we give a point if 
    additional_points = getAdditionalScore(name_results, input("Insert 'year:YYYY' or 'language: Language' to add additional info for the search \n"), df_movies)
    additional_score = {key: name_results.get(key, 0) + additional_points.get(key, 0) for key in set(name_results) | set(additional_points)}
    search_results = search_with_tfidf(query, vocabulary, inverted_index_2, doc_norm, additional_score=additional_score)
    return top_k(search_results, df_movies, top)
    
def search(i, query, top):
    if i == 1:
      search_1(query, top)
    elif i == 2:
      search_2(query, top)
    elif i == 3:
      search_3(query, top)