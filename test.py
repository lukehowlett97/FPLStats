# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:26:52 2023

@author: chcuk
"""

import requests
import json
import pandas as pd

def get_basic_data():

    # Send a HTTP request to the URL
    response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    
    # If the request was successful, the status_code will be 200
    if response.status_code == 200:
        # Get the content of the response
        data = response.json()
    
        # Extract different sections of the data into pandas DataFrames
        elements_df = pd.DataFrame(data['elements'])
        element_types_df = pd.DataFrame(data['element_types'])
        teams_df = pd.DataFrame(data['teams'])
        
        return elements_df, element_types_df, teams_df
    
    else:
        print('Failed to retrieve page, status code:', response.status_code)
    
    
def get_fixtures_data(save = False):
    
    # Send a HTTP request to the URL
    response = requests.get('https://fantasy.premierleague.com/api/fixtures/')
    
    # If the request was successful, the status_code will be 200
    if response.status_code == 200:
        # Get the content of the response
        data = response.json()
    
    else:
        print('Failed to retrieve page, status code:', response.status_code)
        
    if save:
    
        # Convert the list of dictionaries to a JSON string
        json_str = json.dumps(data, indent=4)

        # Write the JSON string to a file
        with open('fixtures.json', 'w') as f:
            f.write(json_str)
            
    return data


elements_df, element_types_df, teams_df = get_basic_data()
fixturesData = get_fixtures_data()


