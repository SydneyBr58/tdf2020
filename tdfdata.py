import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np    
import seaborn as sns
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup


def convert_time(time):
    hours = re.match("[0-9]{2}h", time).group().replace("h", "")
    minutes = re.search("[0-9][0-9](\')", time).group().replace("\'", "")    
    seconds = re.search("[0-9][0-9](\'){2}", time).group().replace("\'\'", "") 
    return int(seconds) + 60*int(minutes) + 3600*int(hours)


def scrape_and_clean(url):
    df = pd.read_html(url)[0]
    df = df[['Coureur', 'Temps']]
    df['Temps'] = df['Temps'].apply(lambda x: convert_time(x))
    df_mod = df.set_index('Coureur')
    df_mod = df_mod.transpose()
    return df_mod


def get_race_details(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    race_str = soup.find_all(class_= "stage-select__option__date")[0].get_text()
    race_str = re.search("[0-9]{2}/[0-9]{2}", race_str).group()
    race_str = race_str.split('/')
    day = int(race_str[0])
    month = int(race_str[1])
    race_date = datetime(year=2020, month=month, day=day).date()
    race_stage = soup.find_all(class_= "stage-select__option__stage")[0].get_text()
    race_route = soup.find_all(class_= "stage-select__option__route")[0].get_text()
    return [race_date, race_stage, race_route]


def prepare_df(df):
    columns = df.columns
    array_df = df.to_numpy()
    for i in range(1,20):
        array_df[i, :] = array_df[i, :] + array_df[i-1, :]  
    df =  pd.DataFrame(array_df, columns=columns)
    return df


def expand_rows(df):
    df.index = (df.index * 10) + 10
    last_idx = df.index[-1] + 1
    df_exp = df.reindex(range(last_idx))
    df_exp['date'] = df_exp['date'].fillna(method='bfill')
    df_exp = df_exp.set_index('date')
    df_rank_exp = df_exp.rank(axis=1, method='first')
    df_exp.iloc[0, df_exp.columns != 'date'] = 0
    df_exp = df_exp.interpolate()
    df_rank_exp = df_rank_exp.interpolate()
    return df_exp, df_rank_exp


def get_all_pages():
    df_all = pd.DataFrame()
    race_days = []
    race_stage = []
    race_route = []
    for i in range(1, 22):
        url = 'https://www.letour.fr/fr/classements/etape-%s' %i
        df = scrape_and_clean(url)
        df_all = df_all.append(df)
        race_days.append(get_race_details(url)[0])
        race_stage.append(get_race_details(url)[1])
        race_route.append(get_race_details(url)[2])
    df_all = prepare_df(df_all)
    df_all['date'] = race_days
    global DICT_STAGE
    DICT_STAGE = dict(zip(race_days, zip(race_stage, race_route)))
    return df_all


def expand_rows(df):
    df.index = (df.index * 10) + 10
    last_idx = df.index[-1] + 1
    df_exp = df.reindex(range(last_idx))
    df_exp['date'] = df_exp['date'].fillna(method='bfill')
    df_exp = df_exp.set_index('date')
    df_rank_exp = df_exp.rank(axis=1, method='first')
    df_exp.iloc[0, df_exp.columns != 'date'] = 0
    df_exp = df_exp.interpolate()
    df_rank_exp = df_rank_exp.interpolate()
    return df_exp, df_rank_exp


def get_data_tdf():
    df = get_all_pages()
    df_plot, df_plot_rank = expand_rows(df)
    return df_plot, df_plot_rank


def create_racer_dict():
    url = 'https://www.letour.fr/fr/classements/etape-1'
    df = df = pd.read_html(url)[0]
    teams = df['Équipe'].drop_duplicates()
    team_color = dict(zip(teams, sns.color_palette("hls", len(teams))))

    class Racer:
        def __init__(self, name, team, color):
            self.name = name
            self.team = team
            self.color = color
        
    racer_dict = {}
    for index, row in df.iterrows():
        racer = Racer(row['Coureur'], row['Équipe'], team_color.get(row['Équipe']))
        racer_dict.update({row['Coureur']: racer})
    return racer_dict


def format_func(sec,pos):
    hours = int(sec//3600)
    minutes = int((sec%3600)//60)
    seconds = int(sec%60)
    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)
   

def plot_chart(j, df, df_rank):
    # Remove the riders who just quitted
    for col in df.columns:
        if df[col].iloc[j] == df[col].iloc[j-1]:
            df = df.drop(columns=col)
            df_rank = df_rank.drop(columns=col)

    ax.clear()
    y = df_rank.iloc[j][df_rank.iloc[j]<21]
    width = df.iloc[j][df_rank.iloc[j] < 21]
    ax.barh(y=y, width=width, color=[RACER_DICT.get(x).color for x in y.index])

    # Show the name and team of each rider
    for i, (time, name, rank) in enumerate(zip(df.iloc[j][df_rank.iloc[j] < 21], 
            df.iloc[j][df_rank.iloc[j] < 21].index, df_rank.iloc[j][df_rank.iloc[j]<21])):
        ax.text(time, rank, name, ha='right', size=8)
        ax.text(time, rank-.25, RACER_DICT.get(name).team, ha='right', size=8)

    # Update the race information
    date = df.iloc[j].name
    legend_text = str(date)+'\n'+str(DICT_STAGE.get(date)[0])+'\n'+str(DICT_STAGE.get(date)[1])
    ax.legend(title=legend_text, bbox_to_anchor=(0.5, -0.15), loc='lower center', fontsize=17)
    ax.set_xlim(0, 320000)
    ax.set_xlabel('Duration (hours)', size=12)
    ax.set_ylabel('Ranking', size=12)

    # formats the x axis as hh:mm:ss and makes a tick every 24 hours
    formatter = FuncFormatter(format_func)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MultipleLocator(base=86400))

    # makes a tick for each int between 1 and 20
    ax.set_yticks(range(1, 21))
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('2020 Tour de France ranking for all stages', size=24, weight=600)
    plt.box(False)


global RACER_DICT
RACER_DICT = create_racer_dict()
fig, ax = plt.subplots(figsize=(15, 15))
df, df_rank = get_data_tdf()
anim = FuncAnimation(fig, plot_chart, init_func=ax.clear,repeat=True,blit=False,frames=len(df), fargs=(df, df_rank),
                             interval=200)

writergif = PillowWriter() 
anim.save('tdf2020.gif', writer=writergif)
plt.show()
