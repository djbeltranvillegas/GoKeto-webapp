import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def dist(center, location):
    R = 3963.19 #miles, assumes spherical earth
    dlon = np.radians(location[1]) - np.radians(center[1])
    dlat = np.radians(location[0]) - np.radians(center[0])
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(center[0])) * np.cos(np.radians(location[0])) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distvar = round(R * c,1)
    return distvar

def fKeto(menid,master_chres2,tf_idf_vector,tfidf_vectorizer_vectors):
    id_document_vector=tf_idf_vector[menid]
    df11 = pd.DataFrame(master_chres2[0], columns=["name"])
    df12 = pd.DataFrame(cosine_similarity(id_document_vector, tfidf_vectorizer_vectors).T, columns=["cossim"])
    df13 = pd.DataFrame(master_chres2[4], columns=["isketo"])
    df14 = pd.DataFrame(master_chres2[3], columns=["carbs"])
    df15 = pd.concat([df11, df12, df13.astype('int64'), df14], axis=1)
    numrec=df15[df15["cossim"] > 0.8]["cossim"].count()
    numketo=df15[(df15["cossim"] > 0.8) & (df15["isketo"] == 1)]["cossim"].count()
    fac=0.0
    if numrec!=0:
        fac=round(float(numketo)/float(numrec),2)
    return fac

#def score_it(carbmax,fatmin):
#    rest_near_list=[]
#    for rest in rest_list:
#        
#    return rest_near_list


def ModelIt(fromUser  = 'Default', births = []):
    in_month = len(births)
    result = in_month
    if fromUser != 'Default':
        return result
    else:
        return 'check your input'