# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:26:52 2021

@author: Lenovo
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'dew':2, 'humidity':9, 'thunder':6,'windspeed':2, 'temp':9, 'pressure':6,'snow':6})

print(r.json())