# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:22:40 2024

@author: chcuk
"""
import json
import requests
import pandas as pd

#plot league gameweek points

#useless doesnt seperate by gw
# league_url = "https://fantasy.premierleague.com/api/leagues-classic/770219/standings/"


def fetch_all_gw_points_data(total_gameweeks, managers):
    all_gw_data = []
    
    for gw in range(1, total_gameweeks + 1):
        gw_data = fetch_gw_points_data(gw, managers)
        all_gw_data.append(gw_data)
        
        print(gw)
        
    return all_gw_data

def fetch_gw_points_data(gameweek, managers):
    
    data = []
    
    for manager in managers:
        manager_name = manager['player_name']
        manager_id = manager['entry']
        # player_name = manager['player_name']
        # team_name = manager['manager_name']

        picks_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gameweek}/picks/"
        picks_response = requests.get(picks_url)

        if picks_response.status_code == 200:
            picks_data = picks_response.json()

            points = picks_data['entry_history']['points']
            points_bench = picks_data['entry_history']['points_on_bench']
            transfer_penalties =picks_data['entry_history']['event_transfers_cost']
            # element_ids = [pick['element'] for pick in picks_data['picks']]
            # captain_id = [pick['element'] for pick in picks_data['picks'] if pick['is_captain']][0]
            
            data.append(
                {'gw': gameweek,
                 'manager': manager,
                 'points': points,
                 'points_on_bench':points_bench,
                 'transfer_penalties': transfer_penalties}
                )
        
    return data
            
def get_managers_data():
    
    # Step 1: Fetch the main JSON for our league
    url = "https://fantasy.premierleague.com/api/leagues-classic/770219/standings/"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching data from {url}. Status code: {response.status_code}")
        exit()

    data = response.json()
    managers = data['standings']['results']
    
    return managers

def format_fetched_data(fetched_data):
    flattened_data = []

    for gw_list in fetched_data:
        
        for manager_dict in gw_list:
            
            md = manager_dict['manager']
            
            manager = md['player_name']
            points = manager_dict['points'] - manager_dict['transfer_penalties']
            gw = manager_dict['gw']
            points_bench = manager_dict['points_on_bench']
        
            flattened_data.append([manager, points, gw, points_bench])
            
    df = pd.DataFrame(flattened_data)
    df.columns = ['manager', 'total_points', 'gw', 'points_bench']

    points_df = df.pivot_table(index= 'manager', columns = 'gw', values='total_points') #.cumsum(axis = 1)
    bench_df = df.pivot_table(index= 'manager', columns = 'gw', values='points_bench') #.cumsum(axis = 1)

    return points_df, bench_df

managers = get_managers_data()

gameweeks = 21

fetched_data = fetch_all_gw_points_data(gameweeks, managers)

df, bench_df = format_fetched_data(fetched_data)

#%%

# plot points on bench

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming 'df' is your DataFrame

# Calculate the total points for each manager
bench_df['Total Points'] = bench_df.sum(axis=1)

# Sort the DataFrame based on the total points
bench_df_sorted = bench_df.sort_values('Total Points', ascending=True)

# Drop the 'Total Points' column
bench_df_sorted = bench_df_sorted.drop('Total Points', axis=1)

# Initialize the left positions for each bar
left_positions = np.zeros(bench_df_sorted.shape[0])

# Create the plot
plt.figure(figsize=(12, 9))
for i, gw in enumerate(bench_df_sorted.columns):
    sns.barplot(y=bench_df_sorted.index, x=bench_df_sorted[gw], label=f'GW {gw}', left=left_positions, color=sns.color_palette('muted', n_colors=len(bench_df_sorted.columns))[i])
    left_positions += bench_df_sorted[gw].values

# Improve the aesthetics
sns.set_style("whitegrid") # or "darkgrid"
sns.despine(left=True, bottom=True)
plt.legend(title='Game Week', fontsize=14)
plt.xlabel('Cumulative Points', fontsize=16)
plt.ylabel('Manager', fontsize=20)
plt.title('Points left on the bench', fontsize=20)
plt.tick_params(axis='y', which='major', labelsize=14)

# Invert the y-axis to have the highest scoring manager at the top
plt.gca().invert_yaxis()

plt.show()


#%%



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Calculate the total points for each manager
df['Total Points'] = df.sum(axis=1)

# Sort the DataFrame based on the total points
df_sorted = df.sort_values('Total Points', ascending=True)

# Drop the 'Total Points' column
df_sorted = df_sorted.drop('Total Points', axis=1)

# Initialize the left positions for each bar
left_positions = np.zeros(df_sorted.shape[0])

# Create the plot
plt.figure(figsize=(12, 9))
for i, gw in enumerate(df_sorted.columns):
    sns.barplot(y=df_sorted.index, x=df_sorted[gw], label=f'GW {gw}', left=left_positions, color=sns.color_palette('muted', n_colors=len(df_sorted.columns))[i])
    left_positions += df_sorted[gw].values

# Improve the aesthetics
sns.set_style("whitegrid") # or "darkgrid"
sns.despine(left=True, bottom=True)
plt.legend(title='Game Week', fontsize=14)
plt.xlabel('Cumulative Points', fontsize=16)
plt.ylabel('Manager', fontsize=20)
plt.title('Manager Performance by Game Week', fontsize=20)
plt.tick_params(axis='y', which='major', labelsize=14)

# Invert the y-axis to have the highest scoring manager at the top
plt.gca().invert_yaxis()

plt.show()


#%%

from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-pastel')

def plot_static_bars(df):
    
    # Sum the points across all game weeks and sort the managers
    total_points = df.sum(axis=1).sort_values(ascending=False)
    sorted_df = df.loc[total_points.index]
    
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Initialize the left array to zero (for the first set of bars)
    left_values = np.zeros(len(sorted_df))
    
    # Loop through each game week and plot
    for gw in sorted_df.columns:
        # Get the values for this game week
        values = sorted_df[gw].values
        # Create the horizontal bar chart, stacking on top of the previous game week
        ax.barh(sorted_df.index, values, left=left_values, label=f'GW {gw}')
        # Update the left values for the next game week
        left_values += values
    
    # Add labels and title
    plt.ylabel('Manager')
    plt.xlabel('Cumulative Points')
    plt.title('Manager Performance by Game Week')
    plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
    
    # Add a legend
    plt.legend(title='Game Week', loc='lower right')  # Adjust legend location as needed
    
    # Invert y-axis to have the manager with the highest points at the top
    ax.invert_yaxis()
    
    # Show the plot
    plt.show()
    
plot_static_bars(df)

#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming 'df' is your DataFrame

# Calculate the total points for each manager
df['Total Points'] = df.sum(axis=1)

# Sort the DataFrame based on the total points
df_sorted = df.sort_values('Total Points', ascending=True)

# Drop the 'Total Points' column
df_sorted = df_sorted.drop('Total Points', axis=1)

# Initialize the left positions for each bar
left_positions = np.zeros(df_sorted.shape[0])

# Create the plot
plt.figure(figsize=(12, 9))
for i, gw in enumerate(df_sorted.columns):
    sns.barplot(y=df_sorted.index, x=df_sorted[gw], label=f'GW {gw}', left=left_positions, color=sns.color_palette('muted', n_colors=len(df_sorted.columns))[i])
    left_positions += df_sorted[gw].values

# Improve the aesthetics
# sns.despine(left=True, bottom=True)

# plt.legend(title='Game Week')
# plt.xlabel('Cumulative Points')
# plt.ylabel('Manager')
# plt.title('Manager Performance by Game Week')

# Improve the aesthetics
sns.set_style("whitegrid") # or "darkgrid"
sns.despine(left=True, bottom=True)
plt.legend(title='Game Week', fontsize=14)
plt.xlabel('Cumulative Points', fontsize=16)
plt.ylabel('Manager', fontsize=20)
plt.title('Manager Performance by Game Week', fontsize=20)
plt.tick_params(axis='y', which='major', labelsize=14)

# Invert the y-axis to have the highest scoring manager at the top
plt.gca().invert_yaxis()

plt.show()


#%%


import bar_chart_race as bcr
import pandas as pd

# Assuming 'df' is your DataFrame prepared for the bar chart race

# Make sure your 'df' is in wide format where:
# - Each row represents a single period (e.g., game week)
# - Each column represents a different manager
# - Each cell contains the value to plot (e.g., points for that game week)

# Example: df = pd.DataFrame(Your Data Here)
df = df.drop(columns = 'Total Points')
df = df.T

df = df.cumsum()


bcr.bar_chart_race(
    df=df,  # Your prepared DataFrame
    filename=  r'C:\Users\chcuk\Work\Projects\fpl\bar_race.mp4',  # Set to None for displaying the animation instead of saving to file
    orientation='h',
    sort='desc',
    # n_bars=10,  # You might want to adjust the number of bars displayed
    fixed_order=False,
    fixed_max=True,
    steps_per_period=30,
    interpolate_period=True,
    label_bars=True,
    bar_size=.95,
    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
    # Change period_fmt if your index is not datetime
    period_fmt=None,  # Set to None or appropriate format string
    period_summary_func=None,  # Adjust or remove if not needed
    perpendicular_bar_func=None,  # Adjust or remove if not needed
    period_length=2000,
    figsize=(5, 3),
    dpi=600,
    cmap='dark12',
    title='Manager Performance by Game Week',
    title_size='',  # Leave empty to use default, or set a specific size
    bar_label_size=7,
    tick_label_size=7,
    # shared_fontdict={'family': 'Helvetica', 'color': '.1'},
    scale='linear',
    writer=None,
    fig=None,
    bar_kwargs={'alpha': .7},
    filter_column_colors=False
    )
    