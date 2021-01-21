from flask import Blueprint, render_template, request, session, g
from flask import current_app
from werkzeug.utils import secure_filename  # 영어파일네임만가능;;
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from my_util.weather import get_weather

rgrs_bp = Blueprint('rgrs_bp', __name__)


def get_weather_main():
    ''' weather = None
    try:
        weather = session['weather']
    except:
        current_app.logger.info("get new weather info")
        weather = get_weather()
        session['weather'] = weather
        session.permanent = True
        current_app.permanent_session_lifetime = timedelta(minutes=60) '''
    weather = get_weather()
    return weather


@rgrs_bp.route('/boston', methods=['GET', 'POST'])
def rg_boston():
    menu = {'ho': 0, 'da': 0, 'ml': 10,
            'se': 0, 'co': 0, 'cg': 0, 'cr': 0, 'wc': 0,
            'cf': 1, 'ac': 0, 're': 0, 'cu': 0}
    boston = load_boston()
    data = boston.data
    label = boston.target
    columns = boston.feature_names
    data = pd.DataFrame(data, columns=columns)
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2, random_state=2021)
    sim_lr = LinearRegression()
    sim_lr.fit(x_train['RM'].values.reshape((-1, 1)), y_train)  # 모델학습
    y_pred = sim_lr.predict(x_test['RM'].values.reshape((-1, 1)))  # 결과예측
    r2 = r2_score(y_test, y_pred)
    weight = sim_lr.coef_[0]
    bias = sim_lr.intercept_
    line_x = np.linspace(np.min(x_test['RM']), np.max(x_test['RM']), 10)
    line_y = sim_lr.predict(line_x.reshape((-1, 1)))
    plt.scatter(x_test['RM'], y_test, s=10, c='black')
    plt.plot(line_x, line_y, c='red')
    plt.legend(['Regression line', 'Test data sample'], loc='upper left')
    img_file = os.path.join(current_app.root_path,
                            'static/img/regression0.png')
    plt.savefig(img_file)  # plt.show대신 이미지저장

    dt_regr = DecisionTreeRegressor(max_depth=5)
    dt_regr.fit(x_train['RM'].values.reshape((-1, 1)), y_train)  # 모델학습
    y_pred = dt_regr.predict(x_test['RM'].values.reshape((-1, 1)))  # 결과예측
    dt_r2 = r2_score(y_test, y_pred)
    line_x = np.linspace(np.min(x_test['RM']), np.max(x_test['RM']), 10)
    line_y = dt_regr.predict(line_x.reshape((-1, 1)))
    plt.scatter(x_test['RM'].values.reshape((-1, 1)), y_test, c='black')
    plt.plot(line_x, line_y, c='red')
    plt.legend(['Regression line', 'Test data sample'], loc='upper left')
    img_file = os.path.join(current_app.root_path,
                            'static/img/regression1.png')
    plt.savefig(img_file)  # plt.show대신 이미지저장

    mtime = int(os.stat(img_file).st_mtime)
    return render_template('regression/boston_res.html', menu=menu, weather=get_weather_main(),
                           r2=r2, weight=weight, bias=bias, dt_r2=dt_r2, mtime=mtime)

    # if request.method == 'GET':
    #     return render_template('regression/boston.html', menu=menu, weather=get_weather())
    # else:
    # test_size_number = int(request.form['test_size_number'])
    # f_csv = request.files['csv']
    # file_csv = os.path.join(current_app.root_path,
    #                         'static/upload/') + f_csv.filename  # 한글파일네임가능
    # f_csv.save(file_csv)
    # current_app.logger.debug(
    #     f"{test_size_number}, {f_csv}, {file_csv}")
    # df_csv = pd.read_csv(file_csv)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     file_csv.data, file_csv.label, f"test_size={test_size_number}, random_state=2021")
    # sim_lr = LinearRegression()
    # sim_lr.fit(x_train['RM'].values.reshape((-1, 1)), y_train)  # 모델학습
    # y_pred = sim_lr.predict(x_test['RM'].values.reshape((-1, 1)))  # 결과예측
    # r2 = r2_score(y_test, y_pred)
    # r2result = r2.format(r2_score(y_test, y_pred))
    # plt.title(f"단순 선형 회귀, R2 : {r2result}")
    # line_x = np.linspace(np.min(x_test['RM']), np.max(x_test['RM']), 10)
    # line_y = sim_lr.predict(line_x.reshape((-1, 1)))
    # plt.scatter(x_test['RM'], y_test, s=10, c='black')
    # plt.plot(line_x, line_y, c='red')
    # plt.legend(['Regression line', 'Test data sample'], loc='upper left')
    # img_file = os.path.join(current_app.root_path,
    #                         'static/img/regression0.png')
    # plt.savefig(img_file)  # plt.show대신 이미지저장
    # mtime = int(os.stat(img_file).st_mtime)  # mtime각각줄필요없이 마지막에 한번만 갱신
    # return render_template('regression/boston_res.html', menu=menu, weather=get_weather_main(),
    #                        test_size_number=test_size_number, r2result=r2result, mtime=mtime)


@rgrs_bp.route('/iris', methods=['GET', 'POST'])
def iris():
    menu = {'ho': 0, 'da': 0, 'ml': 10,
            'se': 0, 'co': 0, 'cg': 0, 'cr': 0, 'wc': 0,
            'cf': 0, 'ac': 0, 're': 1, 'cu': 0}
    if request.method == 'GET':
        return render_template('regression/iris.html', menu=menu, weather=get_weather())
    else:
        index = int(request.form['index'])
        feature_name = request.form['feature']
        column_dict = {'sl': 'Sepal length', 'sw': 'Sepal width',
                       'pl': 'Petal length', 'pw': 'Petal width',
                       'species': ['Setosa', 'Versicolor', 'Virginica']}
        column_list = list(column_dict.keys())

        df = pd.read_csv('static/data/iris_train.csv')
        df.columns = column_list
        X = df.drop(columns=feature_name, axis=1).values
        y = df[feature_name].values

        lr = LinearRegression()
        lr.fit(X, y)
        weight, bias = lr.coef_, lr.intercept_

        df_test = pd.read_csv('static/data/iris_test.csv')
        df_test.columns = column_list
        X_test = df_test.drop(columns=feature_name, axis=1).values[index]
        pred_value = np.dot(X_test, weight.T) + bias

        x_test = list(df_test.iloc[index, :-1].values)
        x_test.append(column_dict['species'][int(df_test.iloc[index, -1])])
        org = dict(zip(column_list, x_test))
        pred = dict(zip(column_list[:-1], [0, 0, 0, 0]))
        pred[feature_name] = np.round(pred_value, 2)
        return render_template('regression/iris_res.html', menu=menu, weather=get_weather(),
                               index=index, org=org, pred=pred, feature=column_dict[feature_name])


nasdaq_dict, kospi_dict, kosdaq_dict = {}, {}, {}  # 기업리스트가 자주바뀌지않으니, 전역변수로만들어놓기


@rgrs_bp.before_app_first_request  # app안에 있어서 그런지 before_app_first_request
def before_app_first_request():
    nasdaq = pd.read_csv('./static/data/NASDAQ.csv', dtype={'Symbol': str})
    for i in nasdaq.index:
        nasdaq_dict[nasdaq['Symbol'][i]] = nasdaq['Name'][i]
    kospi = pd.read_csv('./static/data/KOSPI.csv', dtype={'종목코드': str})
    for i in kospi.index:
        kospi_dict[kospi['종목코드'][i]] = kospi['기업명'][i]
    kosdaq = pd.read_csv('./static/data/KOSDAQ.csv', dtype={'종목코드': str})
    for i in kosdaq.index:
        kosdaq_dict[kosdaq['종목코드'][i]] = kosdaq['기업명'][i]


@rgrs_bp.route('/stock', methods=['GET', 'POST'])
def stock():
    menu = {'ho': 0, 'da': 0, 'ml': 1, 'se': 0, 'co': 0,
            'cg': 0, 'cr': 0, 'st': 1, 'wc': 0, 're': 0}
    if request.method == 'GET':
        return render_template('/regression/stock.html', menu=menu, weather=get_weather(),
                               nasdaq=nasdaq_dict, kospi=kospi_dict, kosdaq=kosdaq_dict)
    else:
        market = request.form['market']
        if market == 'KS':
            code = request.form['kospi_code']
            company = kospi_dict[code]
            code += '.KS'
        elif market == 'KQ':
            code = request.form['kosdaq_code']
            company = kosdaq_dict[code]
            code += '.KQ'
        else:
            code = request.form['nasdaq_code']
            company = nasdaq_dict[code]
        learn_period = int(request.form['learn'])
        pred_period = int(request.form['pred'])  # 클라이언트에게 입력받고
        today = datetime.now()  # 기준이 될 오늘날짜 받고
        start_learn = today - timedelta(days=learn_period*365)  # 기간 정리해서
        end_learn = today - timedelta(days=1)

        try:
            stock_data = pdr.DataReader(
                code, data_source='yahoo', start=start_learn, end=end_learn)  # 야후주식으로부터 실데이터가져오기
            current_app.logger.debug(  # app.logger.debug에서 모듈화하면서 current_app으로변경
                f"get stock data: {code}")  # 터미널에서 확인가능한 개발자확인용
        # model에 적용시키려는 작업(눈으로 확인하려면 주피터노트북에서)
            df = pd.DataFrame({'ds': stock_data.index, 'y': stock_data.Close})
            df.reset_index(inplace=True)
            del df['Date']
        except:
            current_app.logger.error('Date error')  # 야후주식에 데이터가없으면
            flash(f'{company}_{code} 야후주식에 존재하지 않습니다', 'danger')
            return redirect(url_for('stock_bp.stock'))
        try:
            model = Prophet(daily_seasonality=True)  # 학습모델prophet에 적용시키고
            model.fit(df)
        except:
            current_app.logger.error('Value error')  # 야후주식에 데이터가없으면
            flash(f'{company}_{code} 야후주식에 존재하지 않습니다', 'danger')
            return redirect(url_for('stock_bp.stock'))
        future = model.make_future_dataframe(
            periods=pred_period)  # 사용자가 입력한 기간까지 예측한다
        forecast = model.predict(future)

        fig = model.plot(forecast)  # 그래프그리기
        img_file = os.path.join(
            current_app.root_path, 'static/img/stock.png')  # 지정한위치에 지정한파일명으로 저장해라
        fig.savefig(img_file)  # 클라이언트에게 보여주기 위해 파일저장
        mtime = int(os.stat(img_file).st_mtime)  # 이미지저장 즉각반영

        fig = model.plot_components(forecast)  # 시즌분석그래프그리기
        img_file = os.path.join(
            current_app.root_path, 'static/img/stock_plt.png')
        fig.savefig(img_file)
        mtime = int(os.stat(img_file).st_mtime)

        return render_template('/regression/stock_res.html', menu=menu, weather=get_weather(), mtime=mtime, company=company, code=code)
