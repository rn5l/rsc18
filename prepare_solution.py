'''
Created on 09.04.2018

@author: malte
'''

from helper import inout
import pandas as pd


# data folder
FOLDER_TRAIN = 'data/data_formatted_50k/'
FOLDER_TEST = 'data/sample_50k_similar/'
SOLUTION_FILE = 'results_recommender.csv'

if __name__ == '__main__':
    
    solution = pd.read_csv( FOLDER_TEST + SOLUTION_FILE )
    inout.save_submission( FOLDER_TRAIN, solution, FOLDER_TEST + 'submission_' +SOLUTION_FILE, track='main', team='XYZ', contact='x@y.z')
        