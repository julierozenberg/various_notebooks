import pandas as pd
import numpy as np
import prim
import seaborn as sns
from sklearn import mixture

def clustering(n_clusters,data,xstring,ystring,data_norm):
    g = mixture.GaussianMixture(n_components=n_clusters)
    g.fit(data_norm)
    pred = g.predict(data_norm)+1
                    
    x=[]
    y=[]
    for i in range(0,len(data)):
        x.append(data[i][0])
        y.append(data[i][1])

    df = pd.DataFrame()
    df[xstring] = x
    df[ystring] = y
    df['class'] = pred
    
    return df
    
def get_prim(n_clusters, f_value,data,xstring,ystring,inputs,data_norm):
    
    df = clustering(n_clusters,data,xstring,ystring,data_norm)
    classes = df[['class']].drop_duplicates()['class']
    list_prim =[]
    sorted(classes)
        
    for i in range(1,max(classes)+1):
        list_prim.append(prim.Prim(inputs, (df['class']==i),threshold=0.5,threshold_type=">"))

    boxes = []

    for i in range(0,len(list_prim)):
        boxes.append(list_prim[i].find_box())
        obj = (f_value*boxes[i].peeling_trajectory['coverage']-(1-f_value)*boxes[i].peeling_trajectory['density'])**2
        if 1 in boxes[i].peeling_trajectory['coverage']:
            coverage1 = np.where(boxes[i].peeling_trajectory['coverage']==1)[0][0]
            obj = obj.drop(obj.index[[coverage1]])
        k = obj.argmin()
        boxes[i].select(k)
    
    return [boxes,df]

def normalize(data):
    minima = np.min(data, axis=0)
    maxima = np.max(data, axis=0)
    a = 1/(maxima-minima)
    b = minima/(minima-maxima)
    data = a * data + b                    
    return data