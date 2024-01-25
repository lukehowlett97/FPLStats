# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:28:53 2023

@author: chcuk
"""

import requests
import pandas as pd
import time

def fetch_all_gameweeks_data(total_gameweeks, entries):
    all_gw_data = []
    
    for gw in range(1, total_gameweeks + 1):
        gw_data = fetch_gameweek_data(gw, entries)
        all_gw_data.append(gw_data)
        
    return pd.concat(all_gw_data, ignore_index=True)

def fetch_gameweek_data(gameweek, entries):
    all_entries_data = []
    for entry in entries:
        entry_id = entry['entry']
        player_name = entry['player_name']
        team_name = entry['entry_name']

        picks_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gameweek}/picks/"
        picks_response = requests.get(picks_url)

        if picks_response.status_code == 200:
            picks_data = picks_response.json()

            points = picks_data['entry_history']['points']
            element_ids = [pick['element'] for pick in picks_data['picks']]
            captain_id = [pick['element'] for pick in picks_data['picks'] if pick['is_captain']][0]

            entry_data = {
                'player_name': player_name,
                'team_name': team_name,
                'points': points,
                'gameweek': gameweek
            }

            for i in range(15):
                entry_data[f'element_{i+1}'] = element_ids[i]

            entry_data['captain'] = captain_id

            all_entries_data.append(entry_data)
        time.sleep(1)
    return pd.DataFrame(all_entries_data)

# Step 1: Fetch the main JSON
url = "https://fantasy.premierleague.com/api/leagues-classic/770219/standings/"
response = requests.get(url)

if response.status_code != 200:
    print(f"Error fetching data from {url}. Status code: {response.status_code}")
    exit()

data = response.json()
entries = data['standings']['results']

# Fetch data for all gameweeks up to gameweek 6
total_gameweeks = 23
gw_data = fetch_all_gameweeks_data(total_gameweeks, entries)

#%%

# Function to create new column names based on gameweek
def create_gw_columns(column, gameweek):
    if column.startswith('element_'):
        return f'gw{gameweek}_player_{column.split("_")[1]}'
    elif column == 'captain':
        return f'gw{gameweek}_captain'
    elif column == 'points':
        return f'gw{gameweek}_points'
    return column

# Separate the dataframes by gameweek and rename the columns
gw_dfs = []
for gw in gw_data['gameweek'].unique():
    gw_df = gw_data[gw_data['gameweek'] == gw].copy()
    gw_df.drop(columns=['gameweek'], inplace=True)
    gw_df.columns = [create_gw_columns(col, gw) for col in gw_df.columns]
    gw_dfs.append(gw_df)

# Merge the gameweek dataframes on player_name and team_name
final_df = gw_dfs[0]
for gw_df in gw_dfs[1:]:
    final_df = pd.merge(final_df, gw_df, on=['player_name', 'team_name'], how='outer')

# Calculate the total points column
final_df['total_points'] = final_df[[col for col in final_df.columns if 'gw' in col and 'points' in col]].sum(axis=1)

#%%

#convert player ids to names

# Step 1: Fetch the JSON data from the URL and extract player ID to player name mapping
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(url)
json_data = response.json()

# Extracting the mapping of player IDs to player names
player_mapping = {player['id']: player['web_name'] for player in json_data['elements']}

# Step 2: Replace player IDs with player names for each player ID column
# Assuming 'final_df' is the DataFrame from the previous step
for col in final_df.columns:
    if col.startswith("gw") and "player_" in col:
        final_df[col] = final_df[col].map(player_mapping)

# Step 3: Replace captain IDs with player names
for col in final_df.columns:
    if col.startswith("gw") and "captain" in col:
        final_df[col] = final_df[col].map(player_mapping)
        
final_df.to_csv('gw1to6.csv')

#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Update the file path to your specific location
file_path = r"C:\Users\chcuk\Work\Projects\fpl\gw1to6.csv"

# Load the CSV file into a DataFrame
gw_data = pd.read_csv(file_path)

# Group the data by 'player_name' and 'gameweek', then sum the 'points' for each group
grouped_data = gw_data.groupby(['player_name', 'gw']).sum()['points'].reset_index()

# Create a DataFrame to store the cumulative points for each manager
cumulative_data = grouped_data.copy()

# Calculate the cumulative points for each manager across all gameweeks
cumulative_data['cumulative_points'] = grouped_data.groupby('player_name')['points'].cumsum()

# Plot the cumulative line graph
plt.figure(figsize=(20, 10))
sns.lineplot(x='gameweek', y='cumulative_points', hue='player_name', data=cumulative_data)
plt.title('Cumulative Points for Each Manager by Gameweek')
plt.xlabel('Gameweek')
plt.ylabel('Cumulative Points')
plt.legend(title='Manager', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()


#%%

# bootstrap to top players csv

# Importing required libraries
import json
import pandas as pd

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(url)
bootstrap_data = response.json()

# Extracting players data
players_data = bootstrap_data['elements']

# Creating a DataFrame to store the top players' details
top_players_df = pd.DataFrame(columns=['player', 'team', 'position', 'points', 'price', 'points/price'])

# Mapping for position
position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

# Extracting top 15 players for each position
for position in position_mapping.values():
    position_players = [player for player in players_data if player['element_type'] == list(position_mapping.keys())[list(position_mapping.values()).index(position)]]
    position_players_sorted = sorted(position_players, key=lambda x: x['total_points'], reverse=True)[:15]

    for player in position_players_sorted:
        player_name = player['web_name']
        team = bootstrap_data['teams'][player['team'] - 1]['name']
        player_position = position
        points = player['total_points']
        price = player['now_cost'] / 10
        points_per_price = points / price if price != 0 else 0

        top_players_df.loc[len(top_players_df)] = [player_name, team, player_position, points, price, points_per_price]

# Saving the DataFrame to a CSV file
csv_path = "top_players.csv"
top_players_df.to_csv(csv_path, index=False)

top_players_df.head(), csv_path
#%%

a = pd.read_csv(r"C:\Users\chcuk\Downloads\top_players.csv")

mid = a[a['position']=='MID']
