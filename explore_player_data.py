# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:25:15 2024

@author: chcuk
"""


import json
import requests
import pandas as pd


# plot fpl players

def get_player_mapping():

    # Step 1: Fetch the JSON data from the URL and extract player ID to player name mapping
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    json_data = response.json()
    
    # Extracting the mapping of player IDs to player names
    player_mapping = {player['id']: player['web_name'] for player in json_data['elements']}

    return player_mapping

def get_player_points(player, p_id):

    try:

        url = f"https://fantasy.premierleague.com/api/element-summary/{p_id}/"
    
        response = requests.get(url)
        json_data = response.json()
        
        gw_data = json_data['history']
        
        points = []
        
        for gw, gw_dict in enumerate(gw_data):
            points.append({gw+1:gw_dict['total_points']})
    
    except:
        pass
    
    return points

mapping_dict = get_player_mapping()

all_player_points = []

for p_id, p in mapping_dict.items():
    
    all_player_points.append({p:get_player_points(p, p_id)})


#%%
    
def format_player_points(all_player_points, n = 50):
    # Flatten the nested dictionaries
    flattened_data = []
    for player_data in all_player_points:
        for player, points in player_data.items():
            # Flatten the list of dictionaries for each player
            flat_points = {f'gw{week}': point for d in points for week, point in d.items()}
            # Include player name
            flat_points['player'] = player
            # Add to the flattened data list
            flattened_data.append(flat_points)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Reorder columns to have 'player' first
    df = df[['player'] + [col for col in df.columns if col != 'player']]
    
    # Calculate cumulative sum and add as a new column
    df['cumsum'] = df.drop('player', axis=1).sum(axis=1)
    
    df = df.sort_values('cumsum', ascending=False)

    return df.iloc[:n]

def plot_players_line(df):
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'df' is your initial DataFrame
    # Remove 'cumsum' column if it exists
    if 'cumsum' in df.columns:
        df = df.drop(columns='cumsum')
    
    # Set 'player' as index
    df.set_index('player', inplace=True)
    
    # Calculate rolling mean across rows
    rolling_mean = df.rolling(window=5, axis=1).mean()
    
    # Reset index to include 'player' as a column
    rolling_mean = rolling_mean.reset_index()
    
    # Reshape the DataFrame for plotting
    df_melted = rolling_mean.melt(id_vars='player', var_name='gameweek', value_name='points')
    
    # Create the line plot
    plt.figure(figsize=(12, 10))
    sns.lineplot(data=df_melted, x='gameweek', y='points', hue='player', marker='o')
    
    # Improve the aesthetics
    plt.xticks(rotation=45)
    plt.title('Player Rolling Mean Points Across Game Weeks')
    plt.xlabel('Game Week')
    plt.ylabel('Rolling Mean Points')
    plt.legend(title='Player', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

df = format_player_points(all_player_points, 100)

# rolling mean plot
plot_players_line(df)

#%%


import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Assuming df_numeric is your DataFrame
# Exclude 'cumsum' from the features
df.set_index('player', inplace=True)
df_numeric = df.drop(columns=['cumsum'])
df_numeric = df_numeric.fillna(0)

# Prepare the dataset for regression
# Use all columns except the last as features (excluding 'cumsum')
X = df_numeric

# Predictions for each row
predictions = []
model = LinearRegression()
for index, row in X.iterrows():
    # Train the model on all data except the current row
    X_train = X.drop(index)
    y_train = df_numeric.iloc[:, -1].drop(index)

    model.fit(X_train, y_train)

    # Predict the next value for the current row
    prediction = model.predict([row])
    predictions.append(prediction[0])

# Add predictions to the DataFrame
df_numeric['gw22_prediction'] = predictions

# print("Predicted next game week value:", next_gw_prediction[0])



