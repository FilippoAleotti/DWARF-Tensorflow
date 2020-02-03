'''
Configuration helper

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import json
import pprint

def load_settings(settings):
    with open(settings) as json_file:  
        data = json.load(json_file)
    return data

if __name__ == '__main__':
    create_default_settigs()