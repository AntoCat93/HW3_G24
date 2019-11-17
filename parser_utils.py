from bs4 import BeautifulSoup
import pandas as pd 
import os.path


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
            