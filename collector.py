import collector_utils as cu
import json
import pandas as pd

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
headers = {'User-Agent': user_agent}
movies1 = "data//movies1.html"
movies2 = "data//movies2.html"
movies3 = "data//movies3.html"

# Create Dataframes from collector_utils.py
df_movies1 = cu.createDfFromHtml(movies1)
df_movies2 = cu.createDfFromHtml(movies2)
df_movies3 = cu.createDfFromHtml(movies3)

df_movies = pd.concat([df_movies1, df_movies2, df_movies3])
df_movies.reset_index(inplace=True)
df_movies.drop(columns=["index"], inplace=True)

with open('df_movies.json', 'w') as fp:
    json.dump(df_movies.to_json(), fp)
  
## Donwload HTML FILES from collector_utils.py
cu.saveMoviesHtml(df_movies1, user_agent)
cu.saveMoviesHtml(df_movies2, user_agent)
cu.saveMoviesHtml(df_movies3, user_agent)