import pandas as pd
from _collections import deque  #MUST CHECK IF ALLOWED

def get_data(path):
    df = pd.read_csv(path)
    att = df.columns
    objects = df.values.tolist()
    features = att.values.tolist()
    for i in range(0, len(objects)):
        temp = deque(objects[i])
        temp.rotate(-1)
        temp = list(temp)
        objects[i] = temp
    del features[0]
    return objects, features
