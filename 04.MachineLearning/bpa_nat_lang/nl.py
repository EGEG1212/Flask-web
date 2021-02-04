from flask import Blueprint, render_template, request, session, g, flash, redirect, url_for
from flask import current_app
import os
from my_util.weather import get_weather
import json
import requests
import re
import joblib
from urllib.parse import quote
from konlpy.tag import Okt

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

        # 네이버 파파고 번역
        with open('static/keys/papago_key.json') as nkey:
            json_obj = json.load(nkey)
        client_id = list(json_obj.keys())[0]
        client_secret = json_obj[client_id]
        n_url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        n_mapping = {'en': 'en', 'jp': 'ja',
                     'cn': 'zh-CN', 'fr': 'fr', 'es': 'es'}  # kakao:naver
        val = {
            "source": 'ko',
            "target": n_mapping[lang],
            "text": text
        }
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret
        }
        result = requests.post(
            n_url, data=val, headers=headers).json()  # .json()추가
        n_translated_text = result['message']['result']['translatedText']

        # 카카오번역
        with open('static/keys/kakaoaikey.txt') as kfile:
            kai_key = kfile.read(100)
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        k_url = f'https://dapi.kakao.com/v2/translation/translate?query={quote(text)}&src_lang=kr&target_lang={lang}'
        result = requests.get(k_url,
                              headers={"Authorization": "KakaoAK "+kai_key}).json()
        tr_text_list = result['translated_text'][0]
        k_translated_text = '\n'.join(
            [tmp_text for tmp_text in tr_text_list])  # 카카오는 특별히 문장리스트로주기때문에 연결

        # TTS(번역결과읽읽)
        with open('static/keys/papago_key.json') as nkey:
            json_obj = json.load(nkey)
            client_id = list(json_obj.keys())[0]
            client_secret = json_obj[client_id]
        s_mapping = {'en': 'clara', 'jp': 'shinji',
                     'cn': 'liangliang', 'fr': 'matt', 'es': 'jose'}
        speaker = s_mapping[lang]
        speed = "0"
        pitch = "0"
        emotion = "0"
        format = "mp3"
        n_tts_url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"  # 여기 변경
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/x-www-form-urlencoded"}
        val = {
            "speaker": speaker,
            "speed": speed,
            "text": n_translated_text}
        response = requests.post(
            n_tts_url, data=val, headers=headers)  # 위에서처럼 .json()추가했더니자꾸에러ㅋ
        rescode = response.status_code
        if(rescode == 200):
            print(rescode)
            with open('static/voice/cpv_sample_n_translated_text.mp3', 'wb') as f:
                f.write(response.content)

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
            json_obj = json.load(nkey)
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
        text = request.form['text']

        # 카카오 언어감지
        with open('static/keys/kakaoaikey.txt') as kfile:
            kai_key = kfile.read(100)
        k_url = f'https://dapi.kakao.com/v3/translation/language/detect?query={quote(text)}'
        result = requests.get(k_url,
                              headers={"Authorization": "KakaoAK "+kai_key}).json()
        lang = result['language_info'][0]['code']  # 결과는 kr 아니면 en

        # 네이버 파파고
        with open('static/keys/papago_key.json') as nkey:
            json_obj = json.load(nkey)
        client_id = list(json_obj.keys())[0]
        client_secret = json_obj[client_id]
        url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret
        }
        if lang == 'kr':  # 한글이면~
            val = {"source": 'ko', "target": 'en', "text": text}
        else:  # 영어면~
            val = {"source": 'en', "target": 'ko', "text": text}
        result = requests.post(url, data=val, headers=headers).json()
        tr_text = result['message']['result']['translatedText']  # 번역이 된 text

        okt = Okt()
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍',
                     '과', '도', '를', '으로', '자', '에', '와', '한', '하다', '을']
        if lang == 'kr':  # 한글일때, 한글과공백만 남겨두기
            review = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text)
        else:  # 영어라면 tr_text가 한글이겠지? 역시 한글과공백만 남겨두기
            review = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", tr_text)
        morphs = okt.morphs(review, stem=True)  # 토큰화-형태소분석
        ko_review = ' '.join(
            [word for word in morphs if not word in stopwords])  # 불용어 제거,스트링의리스트화['문장'] #append안해도됨?????!!
        # 삼항 lang == 'kr'이면 tr_text가 영어! 아니면 text가 영어!
        en_review = tr_text if lang == 'kr' else text

        naver_tfidf_nb = joblib.load('static/model/naver_tfidf_nb.pkl')
        imdb_tfidf_lr = joblib.load('static/model/imdb_tfidf_lr.pkl')
        pred_ko = '긍정' if naver_tfidf_nb.predict(
            [ko_review])[0] else '부정'  # 만약append했다면 (test_data)[0]
        # ['문장'] 스트링의 리스트로 넣어줘야 predict가되니까 ([ ])[0]이모양
        pred_en = '긍정' if imdb_tfidf_lr.predict([en_review])[0] else '부정'
        # 결론
        if lang == 'kr':
            res = {'src_text': text, 'dst_text': tr_text,
                   'src_pred': pred_ko, 'dst_pred': pred_en}
        else:
            res = {'src_text': text, 'dst_text': tr_text,
                   'src_pred': pred_en, 'dst_pred': pred_ko}

        return render_template('nat_lang/emotion_res.html', res=res,
                               menu=menu, weather=get_weather_main())


# 내마음대로 번역>감정분석 (text의 전처리가빠졌음;)
# @nl_bp.route('/emo', methods=['GET', 'POST'])
# def emo():
#     if request.method == 'GET':
#         return render_template('nat_lang/emo.html', menu=menu, weather=get_weather_main())
#     else:
#         text = request.form['text']
#         trans = request.form['trans']

#         # 카카오언어감지
#         with open('static/keys/kakaoaikey.txt') as kfile:
#             kai_key = kfile.read(100)
#         k_url = 'https://dapi.kakao.com/v3/translation/language/detect?query=' + \
#             quote(text)
#         k_result = requests.get(
#             k_url, headers={"Authorization": "KakaoAK "+kai_key}).json()

#         # 네이버 파파고 번역
#         with open('static/keys/papago_key.json') as nkey:
#             json_str = nkey.read(100)
#         json_obj = json.loads(json_str)
#         client_id = list(json_obj.keys())[0]
#         client_secret = json_obj[client_id]
#         n_url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
#         n_mapping = {'koen': ('ko', 'en'), 'enko': ('en', 'ko')}
#         val = {
#             "source": n_mapping[trans][0],
#             "target": n_mapping[trans][1],
#             "text": text
#         }
#         headers = {
#             "X-NCP-APIGW-API-KEY-ID": client_id,
#             "X-NCP-APIGW-API-KEY": client_secret
#         }
#         result = requests.post(
#             n_url, data=val, headers=headers).json()  # .json()추가
#         n_translated_text = result['message']['result']['translatedText']

#         if k_result['language_info'][0]['code'] == 'kr':
#             # if n_mapping[trans] == 'koen':
#             # 감성분석
#             test_data = []  # ko
#             test_data.append(text)  # (request.form['text'])
#             trans_data = []  # en
#             trans_data.append(n_translated_text)
#             imdb_tfidf_lr = joblib.load('static/model/IMDB_tfidf_lr.pkl')
#             naver_tfidf_nb = joblib.load(
#                 'static/model/naver_tfidf_nb8298.pkl')
#             pred_tl = '긍정' if imdb_tfidf_lr.predict(trans_data)[0] else '부정'
#             pred_tn = '긍정' if naver_tfidf_nb.predict(test_data)[0] else '부정'
#         else:
#             test_data = []  # en
#             test_data.append(text)  # (request.form['text'])
#             trans_data = []  # ko
#             trans_data.append(n_translated_text)
#             imdb_tfidf_lr = joblib.load('static/model/IMDB_tfidf_lr.pkl')
#             naver_tfidf_nb = joblib.load(
#                 'static/model/naver_tfidf_nb8298.pkl')
#             pred_tl = '긍정' if imdb_tfidf_lr.predict(test_data)[0] else '부정'
#             pred_tn = '긍정' if naver_tfidf_nb.predict(trans_data)[0] else '부정'

#         result_dict = {'text': text, 'n_translated_text': n_translated_text,
#                        'pred_tl': pred_tl, 'pred_tn': pred_tn}
#         return render_template('nat_lang/emo_res.html', menu=menu, weather=get_weather_main(),
#                                res=result_dict)
