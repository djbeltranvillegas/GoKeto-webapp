""" 
Following Bootstrap template
"""

from flask import render_template
from app import app
from app.a_model import dist, fKeto
import pandas as pd
from flask import request, send_file
import json
import requests
import numpy as np
import csv
import os
from io import BytesIO
import base64
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import rcParams
import matplotlib.patches as patches

font='DejaVu Sans'

rcParams['axes.linewidth'] = 1
rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 1
rcParams['xtick.minor.size'] = 2
rcParams['xtick.minor.width'] = 1
rcParams['ytick.major.size'] = 4
rcParams['ytick.major.width'] = 1
rcParams['ytick.minor.size'] = 2
rcParams['ytick.minor.width'] = 1
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [font]
rcParams['font.size'] = 16
rcParams['mathtext.default'] = 'regular'
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

from matplotlib import rc # This sometimes gives problems

legend_properties = {'weight':'normal', 'size':14}
axisl_properties = {'family':'sans-serif','sans-serif':[font], 'weight':'bold'}
label_properties = {'weight':'normal', 'size':14}

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
    
    if len(near_rest_list)>0:        
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
        
        
        if len(keto_near_rest_list)>0:
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
                
        else:
            return render_template("nomap.html")
    else:
        return render_template("nomap.html")

@app.route('/kscore')
def kscore_page():
    menu_item = request.args.get('menitem') #gets menu item from previous page
    
    #Open Master Chain Restaurant file: master_chres2
    #Menu_item_name, Total_calories, Calories_from_fat, Total_crabohydrates, Is_it_keto(0 no, 1 yes)
    with open('/home/ubuntu/GoKeto/app/static/data/master_chain_menu.csv') as csvfile3:
        rows3 = csv.reader(csvfile3)
        master_chres2 = list(zip(*rows3))
    
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
    tf_idf_vector=tfidf_vectorizer.transform([menu_item])
    
    id_document_vector=tf_idf_vector[0]
    df11 = pd.DataFrame(master_chres2[0], columns=["name"])
    df12 = pd.DataFrame(cosine_similarity(id_document_vector, tfidf_vectorizer_vectors).T, columns=["cossim"])
    df13 = pd.DataFrame(master_chres2[4], columns=["isketo"])
    df14 = pd.DataFrame(master_chres2[3], columns=["carbs"])
    df16 = pd.DataFrame(master_chres2[1], columns=["totcalo"])
    df17 = pd.DataFrame(master_chres2[2], columns=["calofat"])
    df15 = pd.concat([df11, df12, df13.astype('int64'), df14, df16, df17], axis=1)
    df15["per_calofat"]=np.where((df15['totcalo'].astype('float')>0) & (df15['totcalo'].astype('float')>=df15['calofat'].astype('float')), 100.0*df15["calofat"].astype('float')/df15["totcalo"].astype('float'), np.nan)
    numrec=df15[df15["cossim"] > 0.8]["cossim"].count()
    numketo=df15[(df15["cossim"] > 0.8) & (df15["isketo"] == 1)]["cossim"].count()
    fac=0.0
    if numrec!=0:
        fac=round(float(numketo)/float(numrec),2)
    
    x=df15[df15["cossim"] > 0.8]["carbs"].astype('float').values.tolist()
    y=df15[df15["cossim"] > 0.8]["per_calofat"].astype('float').values.tolist()
    
    kscore_graph = build_graph(x,y)
    
    
    
    return render_template("kscore.html", menu_item=menu_item, numrec=numrec, numketo=numketo, fac=fac, graph1=kscore_graph)


def build_graph(x, y):
    img = BytesIO()
    fig = plt.figure()
    fig.set_size_inches(5.25, 4.00, forward=True)
    ax= fig.add_axes([1/5.25,.75/4,4/5.25,3/4])
    rect = patches.Rectangle((0,70),10,30,linewidth=1,edgecolor='k',facecolor='green')
    ax.add_patch(rect)
    ax.plot(x,y,color='white', markeredgecolor='orange', markeredgewidth=3,linestyle='none',marker='o',markersize=14)
    ax.set_ylabel(r'$\%$ Calories from fat',fontsize=16)
    ax.set_xlabel(r'Carbs, g',fontsize=16)
    if len(x)==0:
        x_up_lim=0.0
    else:
        x_up_lim=1.1*max(x)
    x_right=max(x_up_lim,110)
    ax.set_xlim(0,x_right)
    ax.set_ylim(-5,100)
    ax.set_yticklabels(ax.get_yticks(), fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticks(), fontweight='bold', fontsize=14)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    plt.savefig(img, format='png')
    plt.show()
    #plt.plot(x_coordinates, y_coordinates)
    #plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/kscore.png')
def example1():
    fig = plt.figure()
    fig.set_size_inches(5.25, 4.00, forward=True)
    ax1= fig.add_axes([1/5.25,.75/4,4/5.25,3/4])
    
    #fig, ax = plt.subplots()
    draw1(ax1)
    return nocache(fig_response(fig))

def draw1(ax):
    """Draw a random scatterplot"""
    #x = [random.random() for i in range(100)]
    #y = [random.random() for i in range(100)]
    #ax.scatter(x, y)
    #ax.set_title("Random scatterplot")
    rect = patches.Rectangle((0,70),10,30,linewidth=1,edgecolor='k',facecolor='green')

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.set_ylabel(r'$\%$ Calories from fat',fontsize=16)
    ax.set_xlabel(r'Carbs, g',fontsize=16)
    ax.set_xlim(left=0)
    ax.set_ylim(-5,100)
    ax.set_yticklabels(ax.get_yticks(), fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticks(), fontweight='bold', fontsize=14)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    
    

def fig_response(fig):
    """Turn a matplotlib Figure into Flask response"""
    img_bytes = BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png')
    
def nocache(response):
    """Add Cache-Control headers to disable caching a response"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response


