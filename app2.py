import os
from pathlib import Path
from flask import Flask
from flask import request
from flask import make_response
import io
import csv
import pandas as pd
from flask import render_template, redirect, url_for
from os.path import join, dirname, realpath

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans
import plotly.graph_objs as go

app = Flask(__name__)

@app.route('/')
def index():
   return render_template("index.html")

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

@app.route('/data' , methods=["POST"])
def data():
   
        uploaded_file = request.files['data_file']
        if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
           uploaded_file.save(file_path)
        
        data = pd.read_csv(Path(file_path))
        df2=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
        x = data['Annual Income (k$)']
        y = data['Age']
        z = data['Spending Score (1-100)']
        x = data.iloc[:, [3, 4]].values

# let's check the shape of x
        print(x.shape)
        dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
        plt.title('Dendrogam', fontsize = 20)
        plt.xlabel('Customers')
        plt.ylabel('Ecuclidean Distance')
       # plt.show()


        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(x)
        df2['label']=y_hc
        plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'pink', label = 'miser')
        plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')
        plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')
        plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')
        plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')
        #plt.scatter(hc.cluster_centers_[:,0], hc.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

        plt.style.use('fivethirtyeight')
        plt.title('Hierarchial Clustering', fontsize = 20)
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.legend()
        plt.grid()  
       # plt.show()
        cust1=df2[df2["label"]==1]
        print('Number of customer in 1st group=', len(cust1))
        print('They are -', cust1["CustomerID"].values)
        cust2=df2[df2["label"]==2]
        cust3=df2[df2["label"]==0]
        cust4=df2[df2["label"]==3]
        cust5=df2[df2["label"]==4]
        #return render_template("data.html",df2=df2,l1=len(cust1),c1=cust1["CustomerID"])
        l1=len(cust1)
        l2=len(cust2)
        l3=len(cust3)
        l4=len(cust4)
        l5=len(cust5)
        l=[l1,l2,l3,l4,l5]
        c=[cust1["CustomerID"],cust2["CustomerID"],cust3["CustomerID"],cust4["CustomerID"],cust5["CustomerID"]]        
        return render_template("data.html",l=l,c=c)
if __name__ == '__main__':
   app.run(debug=True)