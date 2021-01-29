from flask import Blueprint, render_template, request, session, g, flash, redirect, url_for
from flask import current_app
from werkzeug.utils import secure_filename  # 영어파일네임만가능;;
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import pandas as pd
import matplotlib.pyplot as plt
from my_util.weather import get_weather

clus_bp = Blueprint('clus_bp', __name__)
menu = {'ho': 0, 'da': 0, 'ml': 1,
        'se': 0, 'co': 0, 'cg': 0, 'cr': 0, 'wc': 0,
        'cf': 0, 'ac': 0, 're': 0, 'cu': 1}


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


@clus_bp.route('/cluster', methods=['GET', 'POST'])
def cluster():
    if request.method == 'GET':
        return render_template('cluster/cluster.html', menu=menu, weather=get_weather_main())
    else:
        k_number = int(request.form['k_number'] or '2')
        try:
            f_csv = request.files['csv']
            file_csv = os.path.join(current_app.root_path,
                                    'static/upload/') + f_csv.filename  # 한글파일네임가능
            f_csv.save(file_csv)
        except:
            current_app.logger.error('no file error')
            flash(
                f'no file error : 파일을 첨부하세요.', 'danger')
            return redirect(url_for('clus_bp.cluster'))
        current_app.logger.debug(f"{k_number}, {f_csv}, {file_csv}")

        df_csv = pd.read_csv(file_csv)
        # 전처리 - 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(
            df_csv.iloc[:, :-1])  # 마지막끝열y타겟값제외하고 st스케일링

        # 차원 축소(PCA)
        pca = PCA(n_components=2)
        pca_array = pca.fit_transform(X_scaled)
        df = pd.DataFrame(pca_array, columns=[
                          'pca_x', 'pca_y'])  # 2차원축소라서 x,y축에
        # 마지막끝열y타겟값만 넣고 화면에그릴 새로운 df완성
        df['target'] = df_csv.iloc[:, -1].values

        # K-Means Clustering
        kmeans = KMeans(n_clusters=k_number, init='k-means++',
                        max_iter=300, random_state=2021)
        kmeans.fit(X_scaled)  # 비지도학습이라 fit만 있음. 스케일된값 X_scaled
        df['cluster'] = kmeans.labels_

        # 시각화
        markers = ['s', 'o', '^', 'P', 'D', 'H', 'x']
        plt.figure()  # plt초기화
        for i in df.target.unique():  # original데이터
            marker = markers[i]
            x_axis_data = df[df.target == i]['pca_x']
            y_axis_data = df[df.target == i]['pca_y']
            plt.scatter(x_axis_data, y_axis_data, marker=marker)
        plt.title('Original Data Visualization by 2 PCA Components')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        img_file = os.path.join(current_app.root_path,
                                'static/img/cluster0.png')
        plt.savefig(img_file)  # plt.show대신 이미지저장

        plt.figure()  # plt초기화
        for i in range(k_number):  # 입력받은 k_number
            try:
                marker = markers[i]
            except:
                current_app.logger.error('index error')
                flash(
                    f'index error : 군집 수(K)는 7까지 입력 가능합니다.', 'danger')
                return redirect(url_for('clus_bp.cluster'))
            x_axis_data = df[df.cluster == i]['pca_x']
            y_axis_data = df[df.cluster == i]['pca_y']
            plt.scatter(x_axis_data, y_axis_data, marker=marker)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title(f'{k_number} Clusters Visualization by 2 PCA Components')
        img_file = os.path.join(current_app.root_path,
                                'static/img/cluster1.png')
        plt.savefig(img_file)  # plt.show대신 이미지저장

        mtime = int(os.stat(img_file).st_mtime)  # mtime각각줄필요없이 마지막에 한번만 갱신
        return render_template('cluster/cluster_res.html', menu=menu, weather=get_weather_main(),
                               k_number=k_number, mtime=mtime)
