from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


app = Flask(__name__)
df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global df
    file = request.files['file']
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    head_df = df.head().to_html()
    return render_template('index.html', head_df=head_df, df_available=True)

@app.route('/describe', methods=['POST'])
def describe():
    global df
    if df is not None:
        description = df.describe().to_html()
        return render_template('index.html', description=description, df_available=True)
    else:
        return render_template('index.html', df_available=False)

@app.route('/visualize', methods=['POST','GET'])
def visualize():
    global df
    description = df.describe().to_html()
    options = list(df.columns)
    return render_template('visualize.html', description=description, df_available=True)

@app.route('/boxplot', methods=['POST'])
def boxplot():
    global df
    if df is not None:
        plt.boxplot(df)
        plt.title('Box Plot')
        
        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()

        return render_template('visualize.html', boxplot=plot_data, df_available=True)
    else:
        return render_template('visualize.html', df_available=False)

@app.route('/heatmap', methods=['POST'])
def heatmap():
    import seaborn as sns
    global df
    import json
    if df is not None:
        sns.heatmap(df)
        plt.title('Heat Map')
        
        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()

        return render_template('visualize.html', heatmap=plot_data, df_available=True)
    else:
        return render_template('visualize.html', df_available=False)
    
@app.route('/scatter', methods=['POST'])
def scatter():
    global df
    if df is not None:
        x_column = request.form.get("x_column")
        y_column = request.form.get("y_column")
        x = df[x_column]
        y = df[y_column]
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.scatter(x,y)
        plt.title("Scatter Plot")

        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()

        return render_template('visualize.html', scatter=plot_data, df_available=True)
    else:
        return render_template('visualize.html', df_available=False)

@app.route('/timeseries', methods=['POST'])
def timeseries():
    global df
    if df is not None:
        df['DATE'] = pd.to_datetime(df['DATE'])
        d1 = df.drop('DATE',axis=1)
        plt.plot(d1)
        plt.title('Timeseries')
        plt.xlabel('Time')
        plt.ylabel('Values')


        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()

        return render_template('visualize.html', timeseries=plot_data, df_available=True)
    else:
        return render_template('visualize.html', df_available=False)

@app.route('/classification', methods=['POST', 'GET'])
def classification():
    global df
    description = df.describe().to_html()
    return render_template('classification.html', description=description, df_available=True)

@app.route('/logistic', methods=['POST'])
def logistic():
    global df
    x = df.iloc[::,0:-1]
    y = df.iloc[::,-1]  
    from sklearn.linear_model import LogisticRegression as lr
    if df is not None:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        classifier = lr()
        model = classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        cm = confusion_matrix(y_test,y_pred) 
        op = f'Classification matrix : \n\n {cm} \n\nAccuracy score : {accuracy_score(y_test,y_pred)}'
        return render_template('classification.html', logistic_result=op, df_available=True)
    else:
        return render_template('classification.html', df_available=False)
    
@app.route('/multiplelogistic', methods=['POST'])
def multiplelogistic():
    global df
    x = df.iloc[::,0:-1]
    y = df.iloc[::,-1]  
    from sklearn.linear_model import LogisticRegression as lr
    if df is not None:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        classifier = lr()
        model = classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        cm = confusion_matrix(y_test,y_pred) 
        op = f'Classification matrix : \n\n {cm} \n\nAccuracy score : {accuracy_score(y_test,y_pred)}'
        return render_template('classification.html', multiplelogistic_result=op, df_available=True)
    else:
        return render_template('classification.html', df_available=False)

@app.route('/decisiontree', methods=['POST'])
def decisiontree():
    global df
    x = df.iloc[::,0:-1]
    y = df.iloc[::,-1]
    if df is not None:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier()
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred) 
        op = f'Classification matrix : \n\n {cm} \n\nAccuracy score : {accuracy_score(y_test,y_pred)}'
        return render_template('classification.html', decisiontree_result=op, df_available=True)
    else:
        return render_template('classification.html', df_available=False)

@app.route('/randomforest', methods=['POST'])
def randomforest():
    global df
    x = df.iloc[::,0:-1]
    y = df.iloc[::,-1]
    if df is not None:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        cm = confusion_matrix(y_test,y_pred) 
        op = f'Classification matrix : \n\n {cm} \n\nAccuracy score : {accuracy_score(y_test,y_pred)}'
        return render_template('classification.html', randomforest_result=op, df_available=True)
    else:
        return render_template('classification.html', df_available=False)

@app.route('/visualizeregression', methods=['POST'])
def visualizeregression():
    global df
    description = df.describe().to_html()
    return render_template('visualizeregression.html', description=description, df_available=True)

@app.route('/linearregression', methods=['POST'])
def linearregression():
    global df
    x = df.iloc[::,1]
    y = df.iloc[::,-1]  
    if df is not None:
        predict = int(request.form.get("predict"))
        from sklearn.linear_model import LinearRegression
        lrregressor = LinearRegression()
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        x_train_values = x_train.values.reshape(-1,1)
        x_test_values = x_test.values.reshape(-1,1)
        y_train_values = y_train.values.reshape(-1,1)
        lrregressor.fit(x_train_values, y_train_values)
        y_pred = lrregressor.predict(x_test_values)
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)
        op = f'Mean Squared Error : {mse} \n\nR2 Score : {r2score}'

        predict1 = np.array(predict)
        x_test_values1 = np.append(x_test_values,predict1)
        x_test_values2 = x_test_values1.reshape(-1,1)
        y_pred1 = lrregressor.predict(x_test_values2)
        x_grid = np.arange(min(x),predict,0.1)
        x_grid = x_grid.reshape((len(x_grid), -1))
        plt.scatter(x_test, y_test, color = 'red')
        plt.scatter(x_test_values2, y_pred1, color = 'green')
        plt.xlabel("Years of experience")
        plt.ylabel("salary")
        plt.legend("green = prediction, red = test")
        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()
        return render_template('visualizeregression.html', linear_result=op,linear_visual_result=plot_data ,df_available=True)
    
    else:
        return render_template('visualizeregression.html', df_available=False)

@app.route('/decisiontreeregression', methods=['POST'])
def decisiontreeregression():
    global df
    x = df.iloc[::,1]
    y = df.iloc[::,-1]  
    if df is not None:
        predict = int(request.form.get("predict"))
        from sklearn.tree import DecisionTreeRegressor
        dtregressor = DecisionTreeRegressor()
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        print(x_train.shape)
        print(y_train.shape)
        print()
        x_train_values = x_train.values.reshape(-1,1)
        x_test_values = x_test.values.reshape(-1,1)
        y_train_values = y_train.values.reshape(-1,1)
        print(x_train_values.shape)
        print(y_train_values.shape)
        dtregressor.fit(x_train_values, y_train_values)
        y_pred = dtregressor.predict(x_test_values)
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)
        op = f'Mean Squared Error : {mse} \n\nR2 Score : {r2score}'

        predict1 = np.array(predict)
        x_test_values1 = np.append(x_test_values,predict1)
        x_test_values2 = x_test_values1.reshape(-1,1)
        y_pred1 = dtregressor.predict(x_test_values2)
        x_grid = np.arange(min(x),predict,0.1)
        x_grid = x_grid.reshape((len(x_grid), -1))
        plt.scatter(x_test, y_test, color = 'red')
        plt.scatter(x_test_values2, y_pred1, color = 'green')
        plt.xlabel("Years of experience")
        plt.ylabel("salary")
        plt.legend("green = prediction, red = test")
        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()
        return render_template('visualizeregression.html', decision_result=op,decision_visual_result=plot_data ,df_available=True)
    
    else:
        return render_template('visualizeregression.html', df_available=False)

@app.route('/randomforestregression', methods=['POST'])
def randomforestregression():
    global df
    x = df.iloc[::,0:1]
    y = df.iloc[::,-1]
    if df is not None:
        predict = int(request.form.get("predict"))
        from sklearn.ensemble import RandomForestRegressor
        rfregressor = RandomForestRegressor()
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        x_train_values = x_train.values.reshape(-1,1)
        x_test_values = x_test.values.reshape(-1,1)
        y_train_values = y_train.values.reshape(-1,1)
        rfregressor.fit(x_train_values, y_train_values)
        y_pred = rfregressor.predict(x_test_values)
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)
        op = f'Mean Squared Error : {mse} \n\nR2 Score : {r2score}'

        predict1 = np.array(predict)
        x_test_values1 = np.append(x_test_values,predict1)
        x_test_values2 = x_test_values1.reshape(-1,1)
        y_pred1 = rfregressor.predict(x_test_values2)
        plt.scatter(x_test, y_test, color = 'red')
        plt.scatter(x_test_values2, y_pred1, color = 'green')
        plt.xlabel("Years of experience")
        plt.ylabel("salary")
        plt.legend("green = prediction, red = test")
        plot_io = BytesIO()
        plt.savefig(plot_io, format='png')
        plot_io.seek(0)
        plot_data = base64.b64encode(plot_io.getvalue()).decode('utf-8')

        plt.clf()
        return render_template('visualizeregression.html', forest_result=op,forest_visual_result=plot_data ,df_available=True)
    
    else:
        return render_template('visualizeregression.html', df_available=False)
     
if __name__ == '__main__':
    app.run(debug=True)