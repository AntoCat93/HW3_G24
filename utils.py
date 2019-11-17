from bs4 import BeautifulSoup
import requests
import re
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

stop_words = set(stopwords.words('english'))

def remove_chars(text):
    return re.sub(r'[\t\n]+?', ' ', text)

def createTsv():
    for file_id in range(1,30001):
        print(file_id)
        file_name = f'html/movie_{file_id}.html'

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
                blocks[header] += remove_chars(element.text)

            if element.name in {'h2','h3'}:
                selected = element.select('span[id]')
                if selected:
                    header = selected[0]['id'].lower()
                    blocks[header] = ''

        for plot_aliase in ['plot_summary', 'premise']:
            if plot_aliase in blocks:
                blocks['plot'] = blocks[plot_aliase]

        if 'plot' not in blocks:
            print("======= ", file_id)
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

                additional_info[prop_name] = remove_chars(prop_val)


        title = soup.select('h1.firstHeading')[0].text.strip()
        film_name = re.sub('\([^\)]*?film[^\)]*?\)\s*?$','',title)
        print(film_name)

        required_fields = ['directed by', 'produced by', 'written by', 'starring', 'music by', 
                           'release date', 'running time', 'country', 'language', 'budget']
        for field in required_fields:
            if field not in additional_info:
                additional_info[field] = 'NA'

        add_info_text = ' \t '.join(additional_info[field] for field in required_fields)

        tsv_file_name = f'tsv/{file_id}.tsv'
        with open(tsv_file_name, 'w') as file:
            content = 'title \t info \t plot \t name \t ' + ' \t '.join(field for field in required_fields) + '\n'
            content += f"{title} \t {blocks['info']} \t {blocks['plot']} \t {film_name} \t {add_info_text}" + '\n'
            file.write(content)
            

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


def words_into_id(words, vocabulary):
    return [vocabulary[w] for w in words]


def search_1(search_input, vocabulary, stop_words, inverted_index):
    processed_input = processWords(search_input, stop_words)
    processed_input_id = [vocabulary[w] for w in processed_input if w in vocabulary]
    results = set.intersection(*[set(inverted_index[term_id]) for term_id in processed_input_id])
    results = [(0, doc_id) for doc_id in results]
    return results


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
    Nt = len(inverted_index_2[word_id])
    return math.log(doc_number/Nt, 10)

def query_tfids(words_ids, inverted_index, doc_number):
    """  Returns tfidfs for query
    """
    counter = Counter(words_ids)
    return [(num) * get_idf(word_id, inverted_index, doc_number) for word_id, num in counter.items()]


def load_inverted_index_1(convert_keys=True):
    #  Load inverted index 1
    with open('inverted_index.json', 'r') as fp:
        if convert_keys:
            return convert_keys_to_int(json.load(fp))
        return json.load(fp)

def load_inverted_index_2(convert_keys=True):
    #  Load inverted index 2
    with open('inverted_index_2.json', 'r') as fp:
        if convert_keys:
            return convert_keys_to_int(json.load(fp))
        return json.load(fp)

def load_doc_norm(convert_keys=True):
    with open('document_norm.json', 'r') as fp:
        if convert_keys:
            return convert_keys_to_int(json.load(fp))
        return json.load(fp)

def load_vocabulary():
    with open('vocabulary.json', 'r') as fp:
        voc = json.load(fp)
    return voc

def convert_keys_to_int(d):
    return {int(key):val for key, val in d.items() }
                       
def filter_documents(processed_input_id, inverted_index):
    """ The function finds all documnets that contains all the words in the query.
    The function maintain cur_index list containing current location of pointer for each list of documents.
    A current index for maximim document id (max_doc_id) stays the same whereas other indexes keep increasing until 
    document id is greater or equal max_doc_id. After that, if all document id are the same, it is added to 
    resulting list.
    
    """
    #  Keep index only for required terms (query terms)
    filtered_index = [inverted_index[term_id] for term_id in processed_input_id]
    
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


def not_strict_filter_documents(processed_input_id, inverted_index):
    """ The function finds all documents that contains at least one query terms.
    It return a list of list of tuples (document_id, tfidf) 
    """
    docs = defaultdict(dict)
    i = -1
    for term_id in processed_input_id:
        i += 1
        for doc_id, tfidf in inverted_index[term_id]:
            docs[doc_id][i] = tfidf
    
    result = []
    for doc_id in docs:
        row = []
        for i in range(len(processed_input_id)):
            if i in docs[doc_id]:
                row.append((doc_id, docs[doc_id][i]))
            else:
                row.append((doc_id, 0))
        result.append(row)
                
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
    return docs_data[0][0], res / (doc_norm[docs_data[0][0]])
    
def search_with_tfidf(search_input, vocabulary, inverted_index, doc_norm, as_heap = True, 
                      strict_terms_filter=True, additional_score = None):
    """  The function transform search input to vocabulary ids, filter documents by search terms and
    calculate cosine similarity. 
    """
    processed_input = processWords(search_input, stop_words)
    processed_input_id = [vocabulary[w] for w in processed_input if w in vocabulary and vocabulary[w] in inverted_index]
    
    if strict_terms_filter:
        filtered_docs = filter_documents(processed_input_id, inverted_index)
    else:
        #  Filter all documents that contain at leats one query term. It used in task 3 for film names
        filtered_docs = not_strict_filter_documents(processed_input_id, inverted_index)
        
    query_data = query_tfids(processed_input_id, inverted_index, len(doc_norm))
    
    result = []
    
    for i in range(len(filtered_docs)):
        data = cosine(filtered_docs[i], query_data, doc_norm)
        doc_id = data[0]
        score = data[1]
        
        #  Additions score for the task 3
        if additional_score and doc_id in additional_score:
            score += additional_score[doc_id]
        
        #  Pusshing element in a min heap with negative score which makes the heap - max heap 
        heappush(result, (-score, doc_id ))

    #  Return dictionaty if as_heap flag is False. It is used in task 3 for film names
    if not as_heap:
        return {doc_id:abs(score) for score, doc_id in result}
        
    return result

def top_k(heap_result, df_movies, k, no_score=False):
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

    #   Show firts 200 characters of intro
    top_res['Intro'] = top_res['Intro'].str.slice(0,200) +' ...'
    
    if no_score:
        top_res.drop(columns=['Score'], inplace=True)

    #  Formating columns so URL can be shown as hyperlink
    top_res = top_res.style.format({'Score': "{:.2f}", 'URL': lambda x: f"<a target=_blank href='{x}'>{x}</a>"})
    
        
    return top_res

