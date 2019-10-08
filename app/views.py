""" 
Following Bootstrap template
"""

from flask import render_template
from app import app
from app.a_model import dist, fKeto
import pandas as pd
from flask import request
import json
import requests
import numpy as np
import csv
import os


# here's the homepage
@app.route('/')
def homepage():
    return render_template("index.html") 

# example page for linking things
@app.route('/map')
def map_page():
    current_add = request.args.get('cur_loc')
    straddp='+'
    straddp=straddp.join(current_add.split())
    api_key=str(os.environ.get('GO_API_KEY'))
    address='https://maps.googleapis.com/maps/api/geocode/json?address='+straddp+'&key='+api_key
    res_ob = requests.get(address)
    x = res_ob.json() 
    lat=x['results'][0]['geometry']['location']['lat']
    lng=x['results'][0]['geometry']['location']['lng']
    center=[lat,lng]
    #These parameters are not used right now
    max_carb=10.0
    min_fat=0.7
    #To test distance function and printing
    location=[40.7435781, -73.9800032]
    distance=dist(center,location)
    
    #Open Master Chain Restaurant file: master_chres2
    #Menu_item_name, Total_calories, Calories_from_fat, Total_crabohydrates, Is_it_keto(0 no, 1 yes)
    with open('/home/ubuntu/GoKeto/app/static/data/master_chain_menu.csv') as csvfile3:
        rows3 = csv.reader(csvfile3)
        master_chres2 = list(zip(*rows3))

    #Open Local Restaurant file: res
    #Latitude, Longitude, Restaurant_name, Menu_item_name    
    with open('/home/ubuntu/GoKeto/app/static/data/res_menufr.csv') as csvfile2:
        rows2 = csv.reader(csvfile2)
        res = list(zip(*rows2))
    
    
    ###ML setup
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    #instantiate CountVectorizer()
    cv=CountVectorizer()
 
    # Word counts for the words Master Chain Restaurant menu items
    word_count_vector=cv.fit_transform(master_chres2[0])
    
    # Setting the vectorizer with TFIDF
    tfidf_vectorizer=TfidfVectorizer(use_idf=True)
 
    # Vectorizer as vectors
    tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(master_chres2[0])
    
    
    #Create list with distances from Local Restaurants to center of map: distarr
    distarr=[]
    for i in range(len(res[0])):
        distarr.append([res[0][i],res[1][i],res[2][i],res[3][i],dist(center,[float(res[0][i]),float(res[1][i])])])
    
    #Create list with only restaurants within 1 mile of center of map: near_rest_list
    #Latitude, Longitude, Restaurant_name, Menu_item_name, Distance_from_center
    near_rest_list=[]
    for i in range(len(res[0])):
        if distarr[i][4]<=1: #1 mile cutoff!!!
            near_rest_list.append(distarr[i])
            
    # Vectorized menu item names from nearby local restaurants
    tf_idf_vector=tfidf_vectorizer.transform([row[3] for row in near_rest_list])
    
    # Create list with Keto Factor (fraction of matching (TFIDF-style) Menu Items in Master Chain Restaurant list 
    # that are Keto): KetoFactor
    KetoFactor=[]

    for i in range(len(near_rest_list)):
        KetoFactor.append(fKeto(i,master_chres2,tf_idf_vector,tfidf_vectorizer_vectors))
    
    #Create list with only restaurants within 1 mile of center of map AND menu items that are Keto (KetoFactor>0): keto_near_rest_list
    #Latitude, Longitude, Restaurant_name, Menu_item_name, Distance_from_center    
    keto_near_rest_list=[]
    for i in range(len(near_rest_list)):
        if KetoFactor[i]>0.0:
            keto_near_rest_list.append([float(near_rest_list[i][0]),float(near_rest_list[i][1]),near_rest_list[i][2],near_rest_list[i][3],near_rest_list[i][4],KetoFactor[i]])
    
    # FIRST OUTPUT LIST!!!!
    # List with local menu items that are within 1 mile, are Keto, and sorted in decreasing KFactor
    # Latitude, Longitude, Restaurant name, Menu item name, distance, Kfactor
    df21 = pd.DataFrame([row[0] for row in keto_near_rest_list], columns=["lat"])
    df22 = pd.DataFrame([row[1] for row in keto_near_rest_list], columns=["lng"])
    df23 = pd.DataFrame([row[2] for row in keto_near_rest_list], columns=["rname"])
    df24 = pd.DataFrame([row[3] for row in keto_near_rest_list], columns=["mname"])
    df25 = pd.DataFrame([row[4] for row in keto_near_rest_list], columns=["distance"])
    df26 = pd.DataFrame([row[5] for row in keto_near_rest_list], columns=["Kscore"])
    df27 = pd.concat([df21, df22, df23, df24, df25, df26], axis=1)

    dish_list=np.array(df27.sort_values(by=["Kscore"],ascending=False)).tolist()
    
    
    # SECOND OUTPUT LIST!!!!
    # List with unique local restaurants, location (lat, lng) number of Keto menu items and list of those items
    # Latitude, Longitude, Restaurant name, Number of Keto items, list of Keto item names
    rest_list=[]

    for restname in df27["rname"].unique():
        #print(type(restname))
        lat=df27[df27["rname"] == str(restname)]['lat'].iloc[0]
        lng=df27[df27["rname"] == str(restname)]['lng'].iloc[0]
        menulist=[]
        mcount=0
        for menuname in df27[df27["rname"] == str(restname)]['mname']:
            menulist.append(menuname)
            mcount+=1
            #print(restname, menuname)
        rest_list.append([lat, lng, restname, str(mcount),menulist])
        
        map_api="https://maps.googleapis.com/maps/api/js?key="+api_key+"&callback=initMap"
    
    return render_template("map.html", center=json.dumps(center), ndishes=len(dish_list), dish_list=dish_list, nrest=json.dumps(len(rest_list)), rest_list=json.dumps(rest_list), map_api=map_api)



