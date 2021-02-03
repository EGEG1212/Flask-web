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
import json
import requests
from urllib.parse import quote

nl_bp = Blueprint('nl_bp', __name__)
menu = {'ho': 0, 'da': 0, 'ml': 1,
        'se': 0, 'co': 0, 'cg': 0, 'cr': 0, 'wc': 0,
        'cf': 0, 'ac': 0, 're': 0, 'cu': 0, 'nl': 1}


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


# 걍혼자해봄ㅋ 한>다국어번역(네이버,카카오), 네이버는TTS음성지원ㅋ.ㅋ
# @nl_bp.route('/lang', methods=['GET', 'POST'])
# def lang():
#     if request.method == 'GET':
#         return render_template('nat_lang/lang.html', menu=menu, weather=get_weather_main())
#     else:
#         test_data = []
#         test_data.append(request.form['text'])
#         feature = request.form['feature']
#         feature_dict = {'en': '영어', 'es': '스페인어',
#                         'fr': '프랑스어', 'vi': '베트남어'}
#         with open('static/keys/papago_key.json') as nkey:
#             json_str = nkey.read(100)
#             json_obj = json.loads(json_str)
#             client_id = list(json_obj.keys())[0]
#             client_secret = json_obj[client_id]
#             url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
#         val = {
#             "source": 'ko',
#             "target": feature,
#             "text": test_data}
#         headers = {
#             "X-NCP-APIGW-API-KEY-ID": client_id,
#             "X-NCP-APIGW-API-KEY": client_secret}
#         response = requests.post(url,  data=val, headers=headers)
#         rescode = response.status_code
#         if rescode == 200:
#             print(response.text)
#         else:
#             print("Error : " + response.text)
#         result = response.json()
#         n_result_text = result['message']['result']['translatedText']

#         # 카카오번역
#         with open('static/keys/kakaoapikey.txt') as kfile:
#             kai_key = kfile.read(100)
#         url = 'https://dapi.kakao.com/v2/translation/translate?query=' + \
#             quote(test_data[0])+'&src_lang=kr&target_lang='+feature
#         k_result_text = requests.get(
#             url, headers={"Authorization": "KakaoAK "+kai_key}).json()
#         k_result_text = k_result_text['translated_text'][0][0]

#         # TTS
#         with open('static/keys/clova_key.json') as nkey:
#             json_str = nkey.read(100)
#             json_obj = json.loads(json_str)
#             client_id = list(json_obj.keys())[0]
#             client_secret = json_obj[client_id]
#         speaker = "nara"
#         speed = "0"
#         pitch = "0"
#         emotion = "0"
#         format = "mp3"
#         url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"  # 여기 변경
#         headers = {
#             "X-NCP-APIGW-API-KEY-ID": client_id,
#             "X-NCP-APIGW-API-KEY": client_secret,
#             "Content-Type": "application/x-www-form-urlencoded"}
#         val = {
#             "speaker": speaker,
#             "speed": speed,
#             "text": n_result_text}
#         response = requests.post(url,  data=val, headers=headers)
#         rescode = response.status_code
#         if(rescode == 200):
#             print(rescode)
#             with open('static/voice/cpv_sample_n_result_text.mp3', 'wb') as f:
#                 f.write(response.content)
#         else:
#             print("Error : " + response.text)

#         result_dict = {
#             'test_data': test_data[0], 'feature': feature_dict[feature], 'n_result_text': n_result_text, 'k_result_text': k_result_text}
#         return render_template('nat_lang/lang_res.html', menu=menu, weather=get_weather_main(),
#                                res=result_dict)

@nl_bp.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'GET':
        return render_template('nat_lang/translate.html', menu=menu, weather=get_weather_main())
    else:
        text = request.form['text']
        lang = request.form['lang']

        # 네이버 파파고
        with open('static/keys/papago_key.json') as nkey:
            json_str = nkey.read(100)
        json_obj = json.loads(json_str)
        client_id = list(json_obj.keys())[0]
        client_secret = json_obj[client_id]
        n_url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        n_mapping = {'en': 'en', 'jp': 'ja',
                     'cn': 'zh-CN', 'fr': 'fr', 'es': 'es'}
        val = {
            "source": 'ko',
            "target": n_mapping[lang],
            "text": text
        }
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret
        }
        result = requests.post(n_url, data=val, headers=headers).json()
        n_translated_text = result['message']['result']['translatedText']

        # 카카오
        with open('static/keys/kakaoaikey.txt') as kfile:
            kai_key = kfile.read(100)
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        k_url = f'https://dapi.kakao.com/v2/translation/translate?query={quote(text)}&src_lang=kr&target_lang={lang}'
        result = requests.get(k_url,
                              headers={"Authorization": "KakaoAK "+kai_key}).json()
        tr_text_list = result['translated_text'][0]
        k_translated_text = '\n'.join([tmp_text for tmp_text in tr_text_list])

        return render_template('nat_lang/translate_res.html', menu=menu, weather=get_weather_main(),
                               org=text, naver=n_translated_text, kakao=k_translated_text, lang=lang)


@nl_bp.route('/tts', methods=['GET', 'POST'])
def tts():
    if request.method == 'GET':
        return render_template('nat_lang/tts.html', menu=menu, weather=get_weather_main())
    else:
        text = request.form['text']
        speaker = request.form['speaker']
        pitch = request.form['pitch']
        speed = request.form['speed']
        volume = request.form['volume']
        emotion = request.form['emotion']

        with open('static/keys/clova_key.json') as nkey:
            json_str = nkey.read(100)
        json_obj = json.loads(json_str)
        client_id = list(json_obj.keys())[0]
        client_secret = json_obj[client_id]

        url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        val = {
            "speaker": speaker, "speed": speed, "text": text,
            "pitch": pitch, "volume": volume, "emotion": emotion
        }
        response = requests.post(url, data=val, headers=headers)
        rescode = response.status_code
        audio_file = os.path.join(current_app.root_path, 'static/img/cpv.mp3')
        if(rescode == 200):
            with open(audio_file, 'wb') as f:
                f.write(response.content)
        mtime = int(os.stat(audio_file).st_mtime)

        return render_template('nat_lang/tts_res.html', menu=menu, weather=get_weather_main(),
                               res=val, mtime=mtime)


@nl_bp.route('/emotion', methods=['GET', 'POST'])
def emotion():
    if request.method == 'GET':
        return render_template('nat_lang/emotion.html', menu=menu, weather=get_weather_main())
    else:
        pass
