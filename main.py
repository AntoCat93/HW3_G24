from utils import *

with open('df_movies.json', 'r') as fp:
    df_movies = pd.read_json(json.load(fp))


vocabulary = load_vocabulary()


inverted_index = load_inverted_index_1()

inverted_index_2 = load_inverted_index_2()
doc_norm = load_doc_norm()
        
name_index = load_from_json('name_index.json')        
name_norm = load_from_json('name_norm.json')


