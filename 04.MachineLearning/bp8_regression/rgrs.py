from flask import Blueprint, render_template, request, session, g, flash, redirect, url_for
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
menu = {'ho': 0, 'da': 0, 'ml': 10,
        'se': 0, 'co': 0, 'cg': 0, 'cr': 0, 'wc': 0,
        'cf': 0, 'ac': 0, 're': 1, 'cu': 0}


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


@rgrs_bp.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'GET':
        return render_template('regression/diabetes.html', menu=menu, weather=get_weather())
    else:
        index = int(request.form['index'] or '0')
        feature = request.form['feature']  # 단일회귀라 feature한값으로 처리
        df = pd.read_csv('static/data/diabetes_train.csv')
        X = df[feature].values.reshape(-1, 1)
        y = df.target.values

        lr = LinearRegression()
        lr.fit(X, y)
        weight, bias = lr.coef_, lr.intercept_
        try:  # 테스트벨류구하자
            df_test = pd.read_csv('static/data/diabetes_test.csv')
            X_test = df_test[feature][index]
        except:
            current_app.logger.error('index error')
            flash(
                f'index error : 입력하신 "{index}"인덱스는 존재하지않습니다. 인덱스 범위를 확인하세요.', 'danger')
            return redirect(url_for('rgrs_bp.diabetes'))
        y_test = df_test.target[index]
        pred = np.round(X_test * weight[0] + bias, 2)  # 소수점둘째자리까지만 round반올림

        # 시각화
        y_min = np.min(X) * weight[0] + bias
        y_max = np.max(X) * weight[0] + bias
        plt.figure()  # grid,legend가 섞일까봐
        plt.scatter(X, y, label='train')
        plt.plot([np.min(X), np.max(X)], [y_min, y_max], 'r', lw=3)
        plt.scatter([X_test], [y_test], c='r', marker='*', s=100, label='test')
        plt.grid()
        plt.legend()
        plt.title(f'Diabetes target vs. {feature}')
        # plt.show()대신에 이미지파일저장. mtime
        img_file = os.path.join(current_app.root_path,
                                'static/img/diabetes.png')
        plt.savefig(img_file)
        mtime = int(os.stat(img_file).st_mtime)

        result_dict = {'index': index,
                       'feature': feature, 'y': y_test, 'pred': pred}
        return render_template('regression/diabetes_res.html', res=result_dict, mtime=mtime,
                               menu=menu, weather=get_weather())


@rgrs_bp.route('/iris', methods=['GET', 'POST'])
def iris():
    if request.method == 'GET':
        return render_template('regression/iris.html', menu=menu, weather=get_weather())
    else:
        index = int(request.form['index'] or '0')
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
        lr.fit(X, y)  # 학습시키면 weight, bias나옴
        weight, bias = lr.coef_, lr.intercept_
        try:
            df_test = pd.read_csv('static/data/iris_test.csv')  # 테스트셋넣어서
            df_test.columns = column_list
            # 컬럼명바꿔주고, 입력받은 feature제외하고, 입력받은[index]한줄만가져오면됨
            X_test = df_test.drop(columns=feature_name, axis=1).values[index]
        except:
            current_app.logger.error('index error')
            flash(
                f'index error : 입력하신 "{index}"인덱스는 존재하지않습니다. 인덱스 범위를 확인하세요.', 'danger')
            return redirect(url_for('rgrs_bp.iris'))
        # .T하면 값이 하나로 구해짐(T말고 weight = weight.reshape(1, -1)해줘도 됨)
        pred_value = np.dot(X_test, weight.T) + bias

        x_test = list(df_test.iloc[index, :-1].values)
        x_test.append(column_dict['species']
                      [int(df_test.iloc[index, -1])])  # 품종명append
        org = dict(zip(column_list, x_test))  # values5개 x_test 묶어서 딕셔너리만들고
        # 예측값 맨끝제외하고 4가지 sl sw pl pw 0제로로만들고
        pred = dict(zip(column_list[:-1], [0, 0, 0, 0]))
        #{'sl':0, 'sw':0, 'pl':0, 'pw':0}
        # 선택받은feature_name 을 넣어준다!!!!!!!!
        pred[feature_name] = np.round(pred_value, 2)
        return render_template('regression/iris_res.html', menu=menu, weather=get_weather(),
                               index=index, org=org, pred=pred, feature=column_dict[feature_name])


@rgrs_bp.route('/boston', methods=['GET', 'POST'])
def boston():
    if request.method == 'GET':
        feature_dict = {'CRIM': '자치시(town)별 1인당 범죄율', 'ZN': '25,000 평방 피트가 넘는 거주지역 토지 비율',
                        'INDUS': '자치시(town)별 비소매 상업지역 토지 비율',
                        'CHAS': '찰스 강(Charles River)에 대한 변수 (강의 경계에 위치하면 1, 그렇지 않으면 0)',
                        'NOX': '10,000,000당 일산화질소 농도', 'RM': '주택 1가구당 평균 방의 수',
                        'AGE': '1940년 이전에 건축된 소유주택 비율', 'DIS': '5개의 보스턴 고용 센터까지의 가중 거리',
                        'RAD': '방사형 고속도로 접근성 지수', 'TAX': '10,000 달러당 재산 세율', 'PTRATIO': '자치시(town)별 학생/교사 비율',
                        'B': '자치시별 흑인 비율', 'LSTAT': '모집단의 하위계층 비율(%)'}
        return render_template('regression/boston.html', feature_dict=feature_dict,
                               menu=menu, weather=get_weather())
    else:
        try:
            index = int(request.form['index'] or '0')
        except:
            current_app.logger.error('index error')
            flash(
                f'index error : 인덱스를 입력하세요.', 'danger')
            return redirect(url_for('rgrs_bp.boston'))
        feature_list = request.form.getlist('key')
        df = pd.read_csv('static/data/boston_train.csv')
        X = df[feature_list].values
        y = df.target.values
        try:
            lr = LinearRegression()
            lr.fit(X, y)
        except:
            current_app.logger.error('feature error')
            flash(
                f'feature error : feature를 선택하세요.', 'danger')
            return redirect(url_for('rgrs_bp.boston'))
        weight, bias = lr.coef_, lr.intercept_
        try:
            df_test = pd.read_csv('static/data/boston_test.csv')
            X_test = df_test[feature_list].values[index, :]
        except:
            current_app.logger.error('index error')
            flash(
                f'index error : 입력하신 "{index}"인덱스는 존재하지 않습니다. 인덱스 범위를 확인하세요.', 'danger')
            return redirect(url_for('rgrs_bp.boston'))
        y_test = df_test.target[index]
        # tmp = lr.predict(X_test.reshape(1,-1))
        pred = np.dot(X_test, weight.T) + bias
        pred = np.round(pred, 2)                    # pred = np.round(tmp[0])

        result_dict = {'index': index,
                       'feature': feature_list, 'y': y_test, 'pred': pred}
        org = dict(zip(df.columns[:-1], df_test.iloc[index, :-1]))
        return render_template('regression/boston_res.html', res=result_dict, org=org,
                               menu=menu, weather=get_weather())


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
            return redirect(url_for('rgrs_bp.stock'))
        try:
            model = Prophet(daily_seasonality=True)  # 학습모델prophet에 적용시키고
            model.fit(df)
        except:
            current_app.logger.error('Value error')  # 야후주식에 데이터가없으면
            flash(f'{company}_{code} 야후주식에 존재하지 않습니다', 'danger')
            return redirect(url_for('rgrs_bp.stock'))
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
