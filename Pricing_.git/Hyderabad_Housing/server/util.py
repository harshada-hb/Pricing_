import json
import pickle
import numpy as np

__locations =None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __locations
    global __data_columns

    with open("./artifact/columns.json","r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open("./artifact/hyderabad_home_prices_model.pickle","rb") as f:
        __model = pickle.load(f)
    print("loading saved artifact...Done")

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("hitech city",1000,3,2))
    print(get_estimated_price("nizampet", 2000,5,4))
    print(get_estimated_price("banjara hills", 1200,3,3))
    print(get_estimated_price("gachibowli", 2200,6,5))