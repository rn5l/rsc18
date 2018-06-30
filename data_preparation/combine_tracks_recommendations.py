'''
Created on 04.04.2018

@author: user01
'''

import pandas as pd
import os

# data folder
FOLDER = "/media/mpd-share/data/recommendations/"

# where to save the result file
RESULT_FILE = "../data/all_tracks_recommendations.csv"

if __name__ == '__main__':
    
    results = pd.DataFrame([])
    
    filenames = os.listdir(FOLDER)
    for filename in filenames:
        if filename.startswith("recos-") and filename.endswith(".csv"):
            fullpath = os.sep.join((FOLDER, filename))
            df_current = pd.read_csv(fullpath, ';')
            results = results.append(df_current)
    
    print('file length:', len(results))      
    print('number of unique tracks:', len(results.track_id.unique()))
    results.to_csv(RESULT_FILE, index= False)