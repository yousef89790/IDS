from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter import messagebox

import os
import hashlib

import pandas as pd

# Read the data

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# visualization for KPI
import time

# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import threading



def KDDCUP99():

    window = Tk()
    window.title("Protect your self")

    window.geometry("1200x750")
    window.minsize("1200", "750")
    window.maxsize("1200", "750")

    
    winFrame = Frame(window, width="1200",height="750",bg="white")
    winFrame.pack()
    winFrame.pack_propagate(0)

    txtFrame = Frame(winFrame, width="1200",height="450",bg="white")
    txtFrame.pack()
    txtFrame.pack_propagate(0)


    def ExitWindow():
        window.quit()

    window.protocol("WM_DELETE_WINDOW", ExitWindow)

    # text=Text(winFrame, width=80, height=15)
    # text.insert(END, "")
    # text.pack()

    # # create a Scrollbar and associate it with txt
    # scrollb = Scrollbar(winFrame, command=text.yview)
    # scrollb.grid(row=0, column=1, sticky='nsew')
    # text['yscrollcommand'] = scrollb.set

    # Add a Scrollbar(horizontal)
    v = Scrollbar(txtFrame, orient='vertical')
    v.pack(side=RIGHT, fill='y')

    # Add a text widget
    text = Text(txtFrame, width=80, height=10,font=("Georgia, 24"), yscrollcommand=v.set)
    text.insert(END, "")
    text.pack()
    # text=Text(win, font=("Georgia, 24"), yscrollcommand=v.set)

    # Add some text in the text widget
    # for i in range(10):
    #     text.insert(END, "Welcome to Tutorialspoint...\n\n")

    # Attach the scrollbar with the text widget
    v.config(command=text.yview)
    text.pack()


    btn1 = Button(winFrame, text="Start", width=16, height=5, command=lambda :threading.Thread(target=get_KDDCUP99).start())
    btn1.pack()


    def insert_text(txt):
        text.insert(END, txt+"\n")
        text.see(END)
    
    def get_KDDCUP99():
        insert_text("KDDCUP99 Start ............")
        # Reading feature list  D:\\Projects\\Antivirus\\Antivirus_KDDCUP99
        # with open("D:\\Projects\\Antivirus\\Antivirus_KDDCUP99\\dataset\\kddcup.names",'r') as f:
        #     print(f.read())
        #     # insert_text(str(f.read()))
        #     text.insert("1.0", f.read())

        with open("D:\\Projects\\Antivirus\\Antivirus_KDDCUP99\\dataset\\kddcup.names", 'r') as f:
            text.insert(INSERT, f.read())
            text.see(END)
        #  Appending columns to the dataset and adding a new column name ‘target’ to the dataset.
        cols = """duration,
        protocol_type,
        service,
        flag,
        src_bytes,
        dst_bytes,
        land,
        wrong_fragment,
        urgent,
        hot,
        num_failed_logins,
        logged_in,
        num_compromised,
        root_shell,
        su_attempted,
        num_root,
        num_file_creations,
        num_shells,
        num_access_files,
        num_outbound_cmds,
        is_host_login,
        is_guest_login,
        count,
        srv_count,
        serror_rate,
        srv_serror_rate,
        rerror_rate,
        srv_rerror_rate,
        same_srv_rate,
        diff_srv_rate,
        srv_diff_host_rate,
        dst_host_count,
        dst_host_srv_count,
        dst_host_same_srv_rate,
        dst_host_diff_srv_rate,
        dst_host_same_src_port_rate,
        dst_host_srv_diff_host_rate,
        dst_host_serror_rate,
        dst_host_srv_serror_rate,
        dst_host_rerror_rate,
        dst_host_srv_rerror_rate"""

        columns = []
        for c in cols.split(','):
            if (c.strip()):
                columns.append(c.strip())
        columns.append('target')

        # print(columns)

        print(len(columns))
        insert_text(str(len(columns)))

        #  Reading the ‘attack_types’ file.

        with open("D:\\Projects\\Antivirus\\Antivirus_KDDCUP99\\dataset\\training_attack_types", 'r') as f:
            text.insert(INSERT, f.read())
            text.see(END)

        # Creating a dictionary of attack_types

        attacks_types = {
            'normal': 'normal',
            'back': 'dos',
            'buffer_overflow': 'u2r',
            'ftp_write': 'r2l',
            'guess_passwd': 'r2l',
            'imap': 'r2l',
            'ipsweep': 'probe',
            'land': 'dos',
            'loadmodule': 'u2r',
            'multihop': 'r2l',
            'neptune': 'dos',
            'nmap': 'probe',
            'perl': 'u2r',
            'phf': 'r2l',
            'pod': 'dos',
            'portsweep': 'probe',
            'rootkit': 'u2r',
            'satan': 'probe',
            'smurf': 'dos',
            'spy': 'r2l',
            'teardrop': 'dos',
            'warezclient': 'r2l',
            'warezmaster': 'r2l',
        }

        # READING DATASET
        # Reading the dataset(‘kddcup.data_10_percent.gz’)
        # and adding Attack Type feature in the training dataset where attack type feature has 5 distinct values i.e. dos, normal, probe, r2l, u2r.

        path = "D:\\Projects\\Antivirus\\Antivirus_KDDCUP99\\dataset\\kddcup.data_10_percent.gz"
        df = pd.read_csv(path, names=columns)

        # Adding Attack Type column

        df['Attack Type'] = df.target.apply(lambda r: attacks_types[r[:-1]])
        df.head()
        # Shape of dataframe and getting data type of each feature
        # The shape of a DataFrame is a tuple of array dimensions that tells the number of rows and columns of a given DataFrame.
        # (494021, 43)
        df.shape
        df['target'].value_counts()
        df['Attack Type'].value_counts()

        df.dtypes
        # DATA PREPROCESSING
        # Finding missing values of all features.
        # No missing value found, so we can further proceed to our next step.

        df.isnull().sum()

        # Finding categorical features
        # Categorical are a Pandas data type. A string variable consisting of only a few different values.
        # Converting such a string variable to a categorical variable will save some memory.
        # The lexical order of a variable is not the same as the logical order (“one”, “two”, “three”).
        num_cols = df._get_numeric_data().columns

        cate_cols = list(set(df.columns)-set(num_cols))
        cate_cols.remove('target')
        cate_cols.remove('Attack Type')

        cate_cols
        # CATEGORICAL FEATURES DISTRIBUTION

        # Visualization graphs
        def bar_graph(feature):
            df[feature].value_counts().plot(kind="bar")
        bar_graph('protocol_type')
        # Protocol type: We notice that ICMP is the most present in the used data, then TCP and almost 20000 packets of UDP type
        plt.figure(figsize=(15, 3))
        bar_graph('service')
        bar_graph('flag')
        bar_graph('logged_in')
        # logged_in (1 if successfully logged in; 0 otherwise): We notice that just 70000 packets are successfully logged in.

        # TARGET FEATURE DISTRIBUTION
        bar_graph('target')

        # Attack Type(The attack types grouped by attack, it's what we will predict)

        bar_graph('Attack Type')
        df.columns
        # DATA CORRELATION

        #  Data Correlation – Find the highly correlated variables using heatmap and ignore them for analysis.
        df = df.dropna('columns')  # drop columns with NaN
        # keep columns where there are more than 1 unique values
        df = df[[col for col in df if df[col].nunique() > 1]]
        corr = df.corr()
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr)
        plt.show()
        df['num_root'].corr(df['num_compromised'])
        df['srv_serror_rate'].corr(df['serror_rate'])
        df['srv_count'].corr(df['count'])
        df['srv_rerror_rate'].corr(df['rerror_rate'])
        df['dst_host_same_srv_rate'].corr(df['dst_host_srv_count'])
        df['dst_host_srv_serror_rate'].corr(df['dst_host_serror_rate'])
        df['dst_host_srv_rerror_rate'].corr(df['dst_host_rerror_rate'])
        df['dst_host_same_srv_rate'].corr(df['same_srv_rate'])
        df['dst_host_srv_count'].corr(df['same_srv_rate'])
        df['dst_host_same_src_port_rate'].corr(df['srv_count'])
        df['dst_host_serror_rate'].corr(df['serror_rate'])
        df['dst_host_serror_rate'].corr(df['srv_serror_rate'])
        df['dst_host_srv_serror_rate'].corr(df['serror_rate'])
        df['dst_host_srv_serror_rate'].corr(df['srv_serror_rate'])
        df['dst_host_rerror_rate'].corr(df['rerror_rate'])

        df['dst_host_rerror_rate'].corr(df['srv_rerror_rate'])
        df['dst_host_srv_rerror_rate'].corr(df['rerror_rate'])

        df['dst_host_srv_rerror_rate'].corr(df['srv_rerror_rate'])
        # This variable is highly correlated with num_compromised and should be ignored for analysis.
        # (Correlation = 0.9938277978738366)
        df.drop('num_root', axis=1, inplace=True)

        # This variable is highly correlated with serror_rate and should be ignored for analysis.
        # (Correlation = 0.9983615072725952)
        df.drop('srv_serror_rate', axis=1, inplace=True)

        # This variable is highly correlated with rerror_rate and should be ignored for analysis.
        # (Correlation = 0.9947309539817937)
        df.drop('srv_rerror_rate', axis=1, inplace=True)

        # This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
        # (Correlation = 0.9993041091850098)
        df.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

        # This variable is highly correlated with rerror_rate and should be ignored for analysis.
        # (Correlation = 0.9869947924956001)
        df.drop('dst_host_serror_rate', axis=1, inplace=True)

        # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
        # (Correlation = 0.9821663427308375)
        df.drop('dst_host_rerror_rate', axis=1, inplace=True)

        # This variable is highly correlated with rerror_rate and should be ignored for analysis.
        # (Correlation = 0.9851995540751249)
        df.drop('dst_host_srv_rerror_rate', axis=1, inplace=True)

        # This variable is highly correlated with dst_host_srv_count and should be ignored for analysis.
        # (Correlation = 0.9865705438845669)
        df.drop('dst_host_same_srv_rate', axis=1, inplace=True)
        df.head()
        df.shape
        df.columns
        df_std = df.std()
        df_std = df_std.sort_values(ascending=True)
        df_std

        # FEATURE MAPPING
        df['protocol_type'].value_counts()
        # Feature Mapping – Apply feature mapping on features such as : ‘protocol_type’ & ‘flag’.

        # protocol_type feature mapping
        pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
        df['protocol_type'] = df['protocol_type'].map(pmap)
        df['flag'].value_counts()
        # flag feature mapping
        fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4,
            'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}
        df['flag'] = df['flag'].map(fmap)
        df.head()
        #  Remove irrelevant features such as ‘service’ before modelling
        df.drop('service', axis=1, inplace=True)
        df.shape
        df.head()
        df.dtypes
        # MODELLING
        # Importing libraries and splitting the datasets

        # from sklearn.model_selection import train_test_split
        # from sklearn.preprocessing import MinMaxScaler
        # from sklearn.metrics import accuracy_score

        df = df.drop(['target',], axis=1)
        print("df.shape")
        print(df.shape)
        insert_text("df.shape")
        insert_text(str(df.shape))
        # Target variable and train set
        Y = df[['Attack Type']]
        X = df.drop(['Attack Type',], axis=1)

        sc = MinMaxScaler()
        X = sc.fit_transform(X)

        # Split test and train data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.33, random_state=42)
        print(X_train.shape, X_test.shape)
        insert_text(str(X_train.shape))
        insert_text(str(X_test.shape))
        print(Y_train.shape, Y_test.shape)
        insert_text(str(Y_train.shape))
        insert_text(str(Y_test.shape))
        text.see(END)

        # Apply various machine learning classification algorithms such as Support Vector Machines,
        # Random Forest, Naive Bayes, Decision Tree, Logistic Regression to create different models.

        # Python implementation of Gaussian Naive Bayes

        # GAUSSIAN NAIVE BAYES
        # Gaussian Naive Bayes
        # from sklearn.naive_bayes import GaussianNB

        print("Gaussian Naive Bayes")
        insert_text("Gaussian Naive Bayes")
        model1 = GaussianNB()
        start_time = time.time()
        model1.fit(X_train, Y_train.values.ravel())
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        text.insert(END, "Training time: ")
        insert_text(str(end_time-start_time))
        start_time = time.time()
        Y_test_pred1 = model1.predict(X_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        text.insert(END, "Testing time: ")
        insert_text(str(end_time-start_time))
        print("Train score is:", model1.score(X_train, Y_train))
        text.insert(END, "Train score is:")
        insert_text(str(model1.score(X_train, Y_train)))
        print("Test score is:", model1.score(X_test, Y_test))
        text.insert(END, "Test score is:")
        insert_text(str(model1.score(X_test, Y_test)))

        print('Report : ')
        print(classification_report(Y_test, Y_test_pred1))
        insert_text("Report :")
        insert_text(str(classification_report(Y_test, Y_test_pred1)))
        # Python implementation of Decision Tree
        # DECISION TREE
        # Decision Tree
        # from sklearn.tree import DecisionTreeClassifier
        print("Decision Tree ")
        insert_text("Decision Tree ")
        model2 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        start_time = time.time()
        model2.fit(X_train, Y_train.values.ravel())
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        text.insert(END, "Training time: ")
        insert_text(str(end_time-start_time))
        start_time = time.time()
        Y_test_pred2 = model2.predict(X_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        text.insert(END, "Testing time: ")
        insert_text(str(end_time-start_time))
        print("Train score is:", model2.score(X_train, Y_train))
        text.insert(END, "Train score is:")
        insert_text(str(model2.score(X_train, Y_train)))
        print("Test score is:", model2.score(X_test, Y_test))
        text.insert(END, "Test score is:")
        insert_text(str(model2.score(X_test, Y_test)))
        print('Report : ')
        print(classification_report(Y_test, Y_test_pred2))
        insert_text("Report :")
        insert_text(str(classification_report(Y_test, Y_test_pred2)))
        # Python code implementation of Random Forest
        # RANDOM FOREST
        # from sklearn.ensemble import RandomForestClassifier
        print("RANDOM FOREST")
        insert_text("RANDOM FOREST")
        model3 = RandomForestClassifier(n_estimators=30)
        start_time = time.time()
        model3.fit(X_train, Y_train.values.ravel())
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        start_time = time.time()
        Y_test_pred3 = model3.predict(X_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        text.insert(END, "Testing time: ")
        insert_text(str(end_time-start_time))
        print("Train score is:", model3.score(X_train, Y_train))
        text.insert(END, "Train score is:")
        insert_text(str(model3.score(X_train, Y_train)))
        print("Test score is:", model3.score(X_test, Y_test))
        text.insert(END, "Test score is:")
        insert_text(str(model3.score(X_test, Y_test)))
        print('Report : ')
        print(classification_report(Y_test, Y_test_pred3))
        insert_text("Report :")
        insert_text(str(classification_report(Y_test, Y_test_pred3)))

        # Python implementation of Support Vector Classifier
        # SUPPORT VECTOR MACHINE
        # from sklearn.svm import SVC
        print("SUPPORT VECTOR MACHINE")
        insert_text("SUPPORT VECTOR MACHINE")
        model4 = SVC(gamma='scale')
        start_time = time.time()
        model4.fit(X_train, Y_train.values.ravel())
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        text.insert(END, "Training time: ")
        insert_text(str(end_time-start_time))
        start_time = time.time()
        Y_test_pred4 = model4.predict(X_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        print("Train score is:", model4.score(X_train, Y_train))
        text.insert(END, "Train score is:")
        insert_text(str(model4.score(X_train, Y_train)))
        print("Test score is:", model4.score(X_test, Y_test))
        print('Report : ')
        print(classification_report(Y_test, Y_test_pred4))
        insert_text("Report :")
        insert_text(str(classification_report(Y_test, Y_test_pred4)))
        # Python implementation of Logistic Regression
        # LOGISTIC REGRESSION
        # from sklearn.linear_model import LogisticRegression
        print("LOGISTIC REGRESSION")
        insert_text("LOGISTIC REGRESSION")
        model5 = LogisticRegression(max_iter=1200000)
        start_time = time.time()
        model5.fit(X_train, Y_train.values.ravel())
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        text.insert(END, "Training time: ")
        insert_text(str(end_time-start_time))
        start_time = time.time()
        Y_test_pred5 = model5.predict(X_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        text.insert(END, "Testing time: ")
        insert_text(str(end_time-start_time))
        print("Train score is:", model5.score(X_train, Y_train))
        text.insert(END, "Train score is:")
        insert_text(str(model5.score(X_train, Y_train)))
        print("Test score is:", model5.score(X_test, Y_test))
        text.insert(END, "Test score is:")
        insert_text(str(model5.score(X_test, Y_test)))
        print('Report : ')
        print(classification_report(Y_test, Y_test_pred5))
        insert_text("Report :")
        insert_text(str(classification_report(Y_test, Y_test_pred5)))
        # Python implementation of Gradient Descent
        # GRADIENT BOOSTING CLASSIFIER
        # from sklearn.ensemble import GradientBoostingClassifier
        print("GRADIENT BOOSTING CLASSIFIER")
        insert_text("GRADIENT BOOSTING CLASSIFIER")
        model6 = GradientBoostingClassifier(random_state=0)
        start_time = time.time()
        model6.fit(X_train, Y_train.values.ravel())
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        text.insert(END, "Training time: ")
        insert_text(str(end_time-start_time))
        start_time = time.time()
        Y_test_pred6 = model6.predict(X_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        text.insert(END, "Testing time: ")
        insert_text(str(end_time-start_time))
        print("Train score is:", model6.score(X_train, Y_train))
        text.insert(END, "Train score is:")
        insert_text(str(model6.score(X_train, Y_train)))
        print("Test score is:", model6.score(X_test, Y_test))
        text.insert(END, "Test score is:")
        insert_text(str(model6.score(X_test, Y_test)))
        print('Report : ')
        print(classification_report(Y_test, Y_test_pred6))
        insert_text("Report :")
        insert_text(str(classification_report(Y_test, Y_test_pred6)))
        # Artificial Neural Network
        # from keras.models import Sequential
        # from keras.layers import Dense
        # from keras.wrappers.scikit_learn import KerasClassifier

        def fun():
            model = Sequential()
            # here 30 is output dimension
            model.add(Dense(30, input_dim=30, activation='relu',
                      kernel_initializer='random_uniform'))
            # in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
            model.add(Dense(1, activation='sigmoid',
                      kernel_initializer='random_uniform'))
            # 5 classes-normal,dos,probe,r2l,u2r
            model.add(Dense(5, activation='softmax'))
            # loss is categorical_crossentropy which specifies that we have multiple classes
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
            return model

        # Since,the dataset is very big and we cannot fit complete data at once so we use batch size.
        # This divides our data into batches each of size equal to batch_size.
        # Now only this number of samples will be loaded into memory and processed.
        # Once we are done with one batch it is flushed from memory and the next batch will be processed.
        model7 = KerasClassifier(build_fn=fun, epochs=100, batch_size=64)
        start = time.time()
        model7.fit(X_train, Y_train.values.ravel())
        end = time.time()
        print('Training time')
        print((end-start))
        start_time = time.time()
        Y_test_pred7 = model7.predict(X_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        start_time = time.time()
        Y_train_pred7 = model7.predict(X_train)
        end_time = time.time()
        accuracy_score(Y_train, Y_train_pred7)
        accuracy_score(Y_test, Y_test_pred7)
        # TRAINING ACCURACY
        names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB', 'ANN']
        values = [87.951, 99.058, 99.997, 99.875, 99.352, 99.793, 98.485]
        f = plt.figure(figsize=(15, 3), num=10)
        plt.subplot(131)
        plt.ylim(80, 102)
        plt.bar(names, values)
        f.savefig('training_accuracy_figure.png', bbox_inches='tight')
        # TESTING ACCURACY
        names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB', 'ANN']
        values = [87.903, 99.052, 99.969, 99.879, 99.352, 99.771, 98.472]
        f = plt.figure(figsize=(15, 3), num=10)
        plt.subplot(131)
        plt.ylim(80, 102)
        plt.bar(names, values)
        f.savefig('test_accuracy_figure.png', bbox_inches='tight')
        names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB', 'ANN']
        values = [1.04721, 1.50483, 11.45332,
            126.96016, 56.67286, 446.69099, 674.12762]
        f = plt.figure(figsize=(15, 3), num=10)
        plt.subplot(131)
        plt.bar(names, values)
        f.savefig('train_time_figure.png', bbox_inches='tight')
        # TESTING TIME
        names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB', 'ANN']
        values = [0.79089, 0.10471, 0.60961,
            32.72654, 0.02198, 1.41416, 0.96421]
        f = plt.figure(figsize=(15, 3), num=10)
        plt.subplot(131)
        plt.bar(names, values)
        f.savefig('test_time_figure.png', bbox_inches='tight')



    window.mainloop()
