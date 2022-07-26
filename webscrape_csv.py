from selenium.webdriver import Chrome
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

import tqdm

import time
import pathlib
from datetime import datetime


def filter_table(text):
    ids = ['round', 'participant', 'time', 'score']
    if text is None:
        return False
    for elem in ids:
        if elem in text:
            return text
    return False


def line2dict(line, season):
    data = dict()
    for elem in line.findAll(class_=filter_table):
        class_type = elem.attrs['class'][1]
        if 'time' in class_type:
            date = elem.get_text(' ')
            date_value = datetime.strptime(date, '%b %d %I:%M %p')
            year = season + (date_value.month < 7)
            data['date'] = date_value.replace(year=year)
        elif 'participant--home' in class_type:
            data['home_team'] = elem.text
        elif 'participant--away' in class_type:
            data['away_team'] = elem.text
        elif 'score--home' in class_type:
            data['home_score'] = elem.text
        elif 'score--away' in class_type:
            data['away_score'] = elem.text
    return data


def open_page(season):
    # driver = Chrome()
    url = f'https://www.flashscore.ca/volleyball/italy/superlega-{season}-{season+1}/results/'

    with Chrome() as driver:
        # Go to website
        driver.get(url)

        # Load all games
        time.sleep(3)
        driver.find_element_by_id('onetrust-accept-btn-handler').click()
        time.sleep(3)

        try:
            while True:
                driver.find_element_by_class_name('event__more').click()
                time.sleep(3)
        except Exception:
            pass

        return driver.page_source


def extract(path=None, season=2017):

    # Create path to save CSV files
    if path is None:
        path = '.'
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Load in BeautifulSoup
    page_source = open_page(season)
    soup = BeautifulSoup(page_source, features='html.parser')

    # Get table
    table = soup.find('div', class_='sportName')

    # Line ids are g_XXXX, where XXXX is some hashing string
    lines = table.findAll('div', id=lambda x: x and x.startswith('g_'))

    # Convert to DataFrame
    df = pd.DataFrame(columns=['date', 'home_team', 'home_score', 'away_team', 'away_score'])
    for i, line in enumerate(lines):
        data = line2dict(line, season)
        df.loc[i] = data

    # Get last date of regular rounds
    for i, line in enumerate(table.contents):
        if 'Round' in line.text:
            round_num = int(line.text.split(' ')[-1])
            if round_num < 10:
                continue
            print(line.text)
            ind = i + 1
            break
    date = table.contents[ind].contents[1].get_text(' ')
    date_value = datetime.strptime(date, '%b %d %I:%M %p')
    year = season + (date_value.month < 7)
    last_date = date_value.replace(year=year)

    # Drop playoffs
    df = df.drop(df[df.date > last_date].index)[::-1].reset_index(drop=True)

    # Save to CSV
    filename = f'SuperLega-{season}-{season+1}.csv'
    df.to_csv(path / filename, index=False)

    return df


path = 'dataset_csv'
for i in tqdm.trange(2009, 2019):
    df = extract(path=path, season=i)

