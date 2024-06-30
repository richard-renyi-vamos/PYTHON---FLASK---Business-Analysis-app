from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 1. Descriptive Statistics
def descriptive_statistics(df):
    return df.describe().to_html()

# 2. Trend Analysis
def trend_analysis(df, date_column, value_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    plt.figure(figsize=(10, 6))
    df[value_column].plot()
    plt.title('Trend Analysis')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plot_path = os.path.join('static', 'trend_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# 3. Correlation Analysis
def correlation_analysis(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plot_path = os.path.join('static', 'correlation_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# 4. Segmentation Analysis
def segmentation_analysis(df, segment_column, value_column):
    segments = df.groupby(segment_column)[value_column].mean().sort_values()
    plt.figure(figsize=(10, 6))
    segments.plot(kind='bar')
    plt.title('Segmentation Analysis')
    plt.xlabel(segment_column)
    plt.ylabel(f'Average {value_column}')
    plot_path = os.path.join('static', 'segmentation_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# 5. Predictive Analysis
def predictive_analysis(df, feature_column, target_column):
    X = df[[feature_column]].values
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.title('Predictive Analysis')
    plt.xlabel(feature_column)
    plt.ylabel(target_column)
    plot_path = os.path.join('static', 'predictive_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return plot_path, mse, r2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        data = pd.read_csv(filepath)
        analysis_type = request.form.get('analysis_type')
        if analysis_type == 'descriptive':
            result = descriptive_statistics(data)
            return render_template('result.html', tables=[result], titles=['Descriptive Statistics'])
        elif analysis_type == 'trend':
            date_column = request.form.get('date_column')
            value_column = request.form.get('value_column')
            plot_path = trend_analysis(data, date_column, value_column)
            return render_template('result.html', plot_path=plot_path)
        elif analysis_type == 'correlation':
            plot_path = correlation_analysis(data)
            return render_template('result.html', plot_path=plot_path)
        elif analysis_type == 'segmentation':
            segment_column = request.form.get('segment_column')
            value_column = request.form.get('value_column')
            plot_path = segmentation_analysis(data, segment_column, value_column)
            return render_template('result.html', plot_path=plot_path)
        elif analysis_type == 'predictive':
            feature_column = request.form.get('feature_column')
            target_column = request.form.get('target_column')
            plot_path, mse, r2 = predictive_analysis(data, feature_column, target_column)
            return render_template('result.html', plot_path=plot_path, mse=mse, r2=r2)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
