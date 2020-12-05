import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
from sklearn.svm import SVC
import base64
from pathlib import Path
import altair as alt

import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# For matrix operations and numerical processing
import numpy as np
import pandas as pd                               # For munging tabular data
# For charts and visualizations
import matplotlib.pyplot as plt
# For displaying images in the notebook
from IPython.display import Image
# For displaying outputs in the notebook
from IPython.display import display
# For labeling Amazon SageMaker models, endpoints, etc.
from time import gmtime, strftime
# For writing outputs to the notebook
import sys
import math                                       # For ceiling function
import json                                       # For parsing hosting outputs
import os                                         # For manipulating filepath names
from sklearn.linear_model import LinearRegression
lin_regression = LinearRegression()

log_regression = LogisticRegression(solver='lbfgs')

DATA_URL = (
    "Cloud_data_service_20201001D35.csv"
)


@st.cache(persist=True)
def load_data(nrows):
    df = pd.read_csv(DATA_URL, nrows=nrows)
    df2 = pd.get_dummies(df['Services'])
    df3 = pd.concat([df, df2], axis=1)
    return df, df2, df3

# Basic preprocessing required for all the models.


def preprocessing(df3, df2, i):

    sample1 = df3[['F1', 'F2', 'F3', 'F4', 'F5', 'F7', df2.columns[i]]]

    # st.write('Service Number at Preprocessing : %s' % (i))

    df_majority = sample1[sample1[df2.columns[i]] == 0]
    df_minority = sample1[sample1[df2.columns[i]] == 1]
    df_majordownsampled = df_majority.sample(df_minority.shape[0])
    sample0 = pd.concat([df_majordownsampled, df_minority])
    Xs = sample0[['F1', 'F2', 'F3', 'F4', 'F5', 'F7']]
    ys = sample0.drop(['F1', 'F2', 'F3', 'F4', 'F5', 'F7'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=0.3, random_state=101)

    return X_train, X_test, y_train, y_test


def svm_model():
    return None


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def cs_header():
    st.markdown('''<img src='data:image/png;base64,{}' class='img-fluid' width=750 height=250>'''.format(
        img_to_bytes("./Cloud-computing-services.png")), unsafe_allow_html=True)
    # img = Image.open("Cloud_computing_services.png")
    # st.image(img, width=300, caption="Simple image")
    return None


def cs_sidebar():
    st.sidebar.header('Usage Guide to adjust values')
    st.sidebar.markdown('__Company size (staff numbers)__')

    st.sidebar.code('''
  1:<10      2:11-50     3:51-100
  4:101-1000 5:1001-3000 6:>3000)
    ''')

    st.sidebar.markdown('__Business sector__')

    st.sidebar.code('''
  1:Innovation & Technology      
  2:Gaming     
  3:Finance
  4:Logistic & Transport 
  5:Media 
  6:Trading & Properties)
    ''')

    st.sidebar.markdown('__Location__')

    st.sidebar.code('''
  1:Local & for Internal use      
  2:Local & for Local Public use    
  3:Local & for Intl use
  4:Regional & for Internal use 
  5:Regional & for Local Public use 
  6:Regional & for Intl use
    ''')

    st.sidebar.markdown('__Annual Revenue (HKD)__')

    st.sidebar.code('''
  1:<50K      2:50K-500K     3:500K-1M
  4:1M-5M     5:5M-10M       6:>10M)
    ''')

    st.sidebar.markdown('__Founding History (Years)__')

    st.sidebar.code('''
  1:<0.5      2:0.5-1     3:1-3
  4:3-10      5:10-20     6:>20)
    ''')

    st.sidebar.markdown('__Number of Systems__')

    st.sidebar.code('''
  1:1         2:2-5       3:6-10
  4:11-20     5:20-50     6:>50)
    ''')

    return None


def cs_body():
    st.title("Cloud Service Classification Algorithms using Streamlit")
    df, df2, df3 = load_data(10000)

    F1 = st.sidebar.slider("Company Size ", 1, 6, 4)
    F2 = st.sidebar.slider("Business Sector ", 1, 6, 5)
    F3 = st.sidebar.slider("Location", 1, 6, 6)
    F4 = st.sidebar.slider("Annual Revenue", 1, 6, 6)
    F5 = st.sidebar.slider("Founding History", 1, 6, 5)
    F7 = st.sidebar.slider("Number of systems", 1, 6, 4)

    if st.sidebar.checkbox("Show RAW data", False):
        st.subheader('Raw Data')
        st.dataframe(df)

    service_count = df2.columns.size
    st.markdown('### There are in total %i services with total of %i records as follows: ' % (
        service_count, df['Services'].size))

    st.write(df['Services'].value_counts())

    servicelist = []
    for aa in range(0, df['Services'].unique().size):
        # st.write('Service Number : %i' % (aa))
        servicelist.append(df2.columns[aa])

    st.markdown(
        'Please select the Cloud service that you would like to evaluate\n')
    choose_svc = st.selectbox("Service to look at", servicelist)

    svc_num = servicelist.index(choose_svc)
    # st.write(svc_num)

    X_train, X_test, y_train, y_test = preprocessing(df3, df2, svc_num)

    if st.sidebar.checkbox("Show training and testing records", False):
        st.subheader('Record distribution')
        # st.dataframe(X_train)
        st.write('Number of X_train records: % i' % (X_train.shape[0]))
        st.write('Number of X_test records: % i' % (X_test.shape[0]))
        st.write('Number of y_train records: % i' % (y_train.shape[0]))
        st.write('Number of y_test records: % i' % (y_test.shape[0]))

    svm = SVC(kernel='rbf', C=10, gamma=0.1)
    svm.fit(X_train, np.array(y_train).reshape(-1))

    knn = KNeighborsClassifier(n_neighbors=26)
    knn.fit(X_train, np.array(y_train).reshape(-1))

    clf = DecisionTreeClassifier(max_depth=10).fit(
        X_train, np.array(y_train).reshape(-1))

    scores = []
    scores.append(svm.score(X_test, y_test))
    scores.append(knn.score(X_test, y_test))
    scores.append(clf.score(X_test, y_test))
    max_pos = 0
    if scores[1] > scores[max_pos]:
        max_pos = 1
    if scores[2] > scores[max_pos]:
        max_pos = 2

    if st.sidebar.checkbox("Show accuracy data", False):
        st.subheader('Accuracy')
        # st.write('Accuracy of SVM classifier on training set: %f' %
        # (svm.score(X_train, y_train)))
        st.write('Accuracy of SVM classifier on test set: %f' %
                 (svm.score(X_test, y_test)))
        # st.write('Accuracy of KNN classifier on training set: %f' %
        # (knn.score(X_train, y_train)))
        st.write('Accuracy of KNN classifier on test set: %f' %
                 (knn.score(X_test, y_test)))
        # st.write('Accuracy of DT classifier on training set: %f' %
        # (clf.score(X_train, y_train)))
        st.write('Accuracy of DT classifier on test set: %f' %
                 (clf.score(X_test, y_test)))
        st.write('Result is %d' % max_pos)

    if max_pos == 0:
        chosen_model = 'SVM'
    elif max_pos == 1:
        chosen_model = 'KNN'
    else:
        chosen_model = 'DT'

    # pos_score = svm.score(X_test, y_test)

    pos_score = scores[max_pos]
    neg_score = 1 - pos_score

    st.write('Chosen model: %s %d' % (chosen_model, max_pos))

    score2_df = pd.DataFrame({
        'accuracy': ['Yes', 'No', ],
        'score': [pos_score, neg_score],
    })
    st.markdown('%s prediction accuracy on service %s' %
                (chosen_model, choose_svc))

    st.write(alt.Chart(data=score2_df, width=500).mark_bar().encode(
        x=alt.X('accuracy', sort=None),
        y='score',
    ))

    # pred = svm.predict(X_test)

    AX_record = [[F1, F2, F3, F4, F5, F7]]
    if max_pos == 0:
        pred = svm.predict(X_test)
        Ay_record = svm.predict(AX_record)
    elif max_pos == 1:
        pred = knn.predict(X_test)
        Ay_record = knn.predict(AX_record)
    else:
        pred = clf.predict(X_test)
        Ay_record = clf.predict(AX_record)

    if st.sidebar.checkbox("Show the confusion matrix", False):
        st.subheader('%s Confusion Matrix' % (chosen_model))
        st.write(confusion_matrix(y_test, pred))

    # st.write('Ans is %i' % Ay_record[0])

    if Ay_record == 0:
        st.markdown(
            '__Sorry !! This customer WILL NOT adopt [%s] Service in near future...__' % (choose_svc))
    else:
        st.markdown(
            '__Great !! This customer is likely going to adopt [%s] Service__' % (choose_svc))

    # st.write('THis customer will likely %s use the Service [%s] in near future' % (YesNo, choose_svc))

    return None


def cs_footer():
    return None


def main():
    cs_header()
    cs_sidebar()
    cs_body()
    cs_footer()
    return None


if __name__ == "__main__":
    main()
