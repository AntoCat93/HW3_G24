from bs4 import BeautifulSoup
import requests
import re
import os.path
import pandas as pd
import time
from time import sleep
import random
import io 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import defaultdict
from collections import Counter
import json
import os.path
import math
from heapq import *
from IPython.core.display import HTML

def createTsv():
  for file_id in range(1,30001):
      file_name = f'movies/movie_{file_id}.html'
      
      if not os.path.isfile(file_name):
          continue
      
      with open(file_name, 'r') as file:
          content = file.read()

      soup = BeautifulSoup(content, 'html.parser')

      header = 'info'
      blocks = {'info' : ''}


      # Info - Plot secction
      for element in soup.select('div.mw-parser-output > *'):
          if element.name == 'p':
              blocks[header] += element.text

          if element.name in {'h2','h3'}:
              selected = element.select('span[id]')
              if selected:
                  header = selected[0]['id'].lower()
                  blocks[header] = ''

      for plot_aliase in ['plot_summary', 'premise']:
          if plot_aliase in blocks:
              blocks['plot'] = blocks[plot_aliase]
          
      if 'plot' not in blocks:
          continue
          
      # Additional info section
      additional_info = {}
      for element in soup.select('.infobox.vevent tr'):
          prop_name = element.find('th')
          prop_val = element.find('td')
          if prop_name and prop_val:
              prop_name = prop_name.get_text().lower()
              # Replace tags by space. If we use 'get_text' some content will be merged without space.
              prop_val = re.sub(r'<[^>]+?>',' ',str(prop_val))
              # Remove space duplicates
              prop_val = re.sub(r'\s+',' ',str(prop_val)).strip()

              additional_info[prop_name] = prop_val


      title = soup.select('h1.firstHeading')[0].text.strip()
      film_name = re.sub('\([^\)]*?film[^\)]*?\)\s*?$','',title)

      required_fields = ['directed by', 'produced by', 'written by', 'starring', 'music by', 'release date', 'running time', 'country', 'language', 'budget']
      for field in required_fields:
          if field not in additional_info:
              additional_info[field] = 'NA'

      add_info_text = ' \t '.join(additional_info[field] for field in required_fields)

      tsv_file_name = f'tsv/{file_id}.tsv'
      with open(tsv_file_name, 'w') as file:
          content = f"{title}\t {blocks['info']} \t {blocks['plot']} \t {film_name} \t {add_info_text}"
          file.write(content)
      #break

def createDfFromHtml(movies):
    with open(movies, encoding="utf-8") as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')
    table_rows = soup.findAll('tr')
    l = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [tr.text for tr in td]
        l.append(row)
    df = pd.DataFrame(l, columns=["ID", "URL"])
    df = df.drop(0)
    return df
    
def saveMoviesHtml(df_movies, headers):
    for index, el in df_movies.iterrows():
        resp = requests.get(el["URL"], headers)
        if resp.status_code == 200:
            open('movies//movie_'+el["ID"]+'.html', 'wb+').write(resp.content)

def processWords(line, stop_words):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(line)
    new_words = set(words) - stop_words
    ps = PorterStemmer()
    stem_words = []
    for w in new_words:
        stem_words.append(ps.stem(w))
    #sorted_words = sorted(stem_words)
    return stem_words

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

def words_into_id(words, vocabulary):
    return [vocabulary[w] for w in words]

def createInvertedIndex(vocabulary, stop_words):
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

def search_1(search_input, vocabulary, stop_words, inverted_index):
    processed_input = processWords(search_input, stop_words)
    processed_input_id = [vocabulary[w] for w in processed_input if w in vocabulary]
    print(processed_input)
    print(processed_input_id)
    return set.intersection(*[set(inverted_index[str(term_id)]) for term_id in processed_input_id])


def text_from_tsv(file_name, col_names, as_text = True):
    """  Returns concatanated (by col_names) text from .tsv file
    """
    df = pd.read_csv(file_name, sep=' \t ', dtype=str, engine='python') #  there is a warning without engine='python'
    data = [df.iloc[0][name] for name in col_names]
    if as_text:
        return ' '.join(data)
    else:
        return data

def get_blocks(line, ids):
    """ The function takes the second line of tsv file( with ' \t ' as seperator) as argument 
    along with ids (title - 0, into - 1, plot - 3 and so on...).
    It returns the list of values of title, into etc. It is written in order to speed up 'data_from_tsv' function 
    which, in turn, speed up index creation and the search output.
    """
    pos_from = 0
    counter = -1
    i = 0
    res = []
    while True:
        pos_to = line.find(' \t ', pos_from)
        counter += 1

        if pos_to == -1:
            break

        if not ids or counter == ids[i]:
            res.append(line[pos_from: pos_to])
            i += 1
        pos_from = pos_to + 3

        if i == len(ids):
            break

    if not ids or (i < len(ids) and counter == ids[i]):
        res.append(line[pos_from: ])

    return res
    
def data_from_tsv(file_name, col_ids, as_text = True):
    """  Returns blocks (title, intro etc.) of text by their number starting from zero.
    If as_text is set True returns concatinated string (used in indexing).
    """
    with open(file_name, 'r') as file:
        line = file.readlines()[1]
        data = get_blocks(line, col_ids)
        if as_text:
            return ' '.join(data)
        else:
            return data

def get_idf(word_id, inverted_index_2, doc_number):
    Nt = len(inverted_index_2[str(word_id)])
    return math.log(doc_number/Nt, 10)

def query_tfids(words_ids, inverted_index, doc_number):
    """  Returns tfidfs for query
    """
    counter = Counter(words_ids)
    return [(num) * get_idf(word_id, inverted_index, doc_number) for word_id, num in counter.items()]

def load_inverted_index_2():
    #  Load inverted index 2
    with open('inverted_index_2.json', 'r') as fp:
        return json.load(fp)

def load_doc_norm():
    with open('document_norm.json', 'r') as fp:
        return json.load(fp)

def load_vocabulary():
    with open('vocabulary.json', 'r') as fp:
        return json.load(fp)
        
def filter_documents(processed_input_id, inverted_index):
    """ The function finds all documnets that contains all the words in the query.
    The function maintain cur_index list containing current location of pointer for each list of documents.
    A current index for maximim document id (max_doc_id) stays the same whereas other indexes keep increasing until 
    document id is greater or equal max_doc_id. After that, if all document id are the same, it is added to 
    resulting list.
    
    """
    #  Keep index only for required terms (query terms)
    filtered_index = [inverted_index[str(term_id)] for term_id in processed_input_id]
    
    #  Get the max document id of first elements of filtered index
    max_doc_id = max([r[0] for r in filtered_index])[0]
    
    result = []
    cur_index = {i:0 for i in range(len(filtered_index))} #  List of indexies for each term (word)
    while True:
        #  Here we store documents for cur_index pointer in order to find max in each iteration
        first_docs = []  
        for i in range(len(filtered_index)):
            
            #  Increaseing cur_index while document id is less than max id 
            while cur_index[i] < len(filtered_index[i]) and filtered_index[i][cur_index[i]][0] < max_doc_id:
                cur_index[i] += 1
            
            #  If index is out of element number exit from loop
            if cur_index[i] == len(filtered_index[i]):
                return result
            
            first_docs.append(filtered_index[i][cur_index[i]])
        
        #  If min == max then all elements are equal which means we find a document that contains all query terms
        if min(first_docs)[0] == max(first_docs)[0]:
            result.append(first_docs)
            for i in range(len(filtered_index)):
                cur_index[i] += 1
        else:
            max_doc_id = max(first_docs)[0]

    return result
        

def cosine(docs_data, query_data, doc_norm):
    """  The function returns cosine for two vectors: document, query.
    """
    res = 0
    doc_len = 0
    query_len = 0
    
    for i in range(len(query_data)):
        res += docs_data[i][1] * query_data[i]
        doc_len += docs_data[i][1] ** 2
        query_len += query_data[i] ** 2
    
    #  Don't need to divide on query norm since it is the same for all documents
    return docs_data[0][0], res / (doc_norm[str(docs_data[0][0])])
    
def search_with_tfidf(search_input, inverted_index, doc_norm, stop_words, vocabulary):
    """  The function transform search input to vocabulary ids, filter documents by search terms and
    calculate cosine similarity. 
    """
    doc_number = 30001
    processed_input = processWords(search_input, stop_words)
    processed_input_id = [vocabulary[w] for w in processed_input if w in vocabulary]
    filtered_docs = filter_documents(processed_input_id, inverted_index)
    query_data = query_tfids(processed_input_id, inverted_index, doc_number)
    
    result = []
    
    for i in range(len(filtered_docs)):
        data = cosine(filtered_docs[i], query_data, doc_norm)
        doc_id = data[0] 
        
        #  Pusshing element in a min heap with negative score which makes the heap - max heap 
        heappush(result, (-data[1], doc_id ))
        
    return result

def top_k(heap_result, k, df_movies, as_html = True):
    """ Returns the top-k or all results, if k is greater than total number of result, in dataframe 
    """
    top_res = []
    for i in range(min(k,len(heap_result))):
        #  Popping the element with the lowest rank (which in fact has the highes rank for us)
        raw = heappop(heap_result)
        score = abs(raw[0])
       
        doc_id = raw[1]
        file_name = f'tsv/{doc_id}.tsv'
        title, intro = data_from_tsv(file_name, [0, 1], as_text = False)
        
        top_res.append((score, title, intro, df_movies.iloc[doc_id-1]['URL']))
    
    top_res = pd.DataFrame(top_res)
    top_res.columns = ['Score', 'Title', 'Intro', 'URL']
    if as_html: 
        #   Show firts 200 characters of intro
        top_res['Intro'] = top_res['Intro'].str.slice(0,200) +' ...'
        
        #  Formating columns so URL can be shown as hyperlink
        top_res = top_res.style.format({'Score': "{:.2f}", 'URL': lambda x: f"<a target=_blank href='{x}'>{x}</a>"})
        
    return top_res

def search_2(query, top, inverted_index_2, stop_words, df_movies):
    vocabulary = load_vocabulary()
    doc_norm = load_doc_norm()
    return top_k(search_with_tfidf(query, inverted_index_2, doc_norm, stop_words, vocabulary), top, df_movies)

def create_inverted_index_2(fields, norm_file=None, index_file=None):
    inverted_index_2 = defaultdict(list)
    i = 1
    doc_number = 10000
    doc_norm = defaultdict(list)

    for file_id in range(1, doc_number):
        fname = f'data/{file_id}.tsv'
        if os.path.isfile(fname):
            text = data_from_tsv(fname, fields)

            #  Extract usefull terms from text 
            words = processWords(text, stop_words)

            #  Convert text represantion of words into ids of vacabulary
            words = words_into_id(words)

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
            inverted_index_2[word_id][i][1] *= get_idf(word_id,inverted_index_2, doc_number)
            inverted_index_2[word_id][i] = tuple(inverted_index_2[word_id][i])

    if index_file:
        #  Store inverted index as json
        with open(index_file, 'w') as fp:
            json.dump(inverted_index_2, fp)

    return inverted_index_2, doc_norm
