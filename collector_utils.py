from bs4 import BeautifulSoup
import requests
import pandas as pd 

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
