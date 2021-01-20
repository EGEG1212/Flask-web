import requests
import xmltodict
import json
from pprint import pprint


def get_corona_data(startCreateDt, endCreateDt):
    url = "http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19SidoInfStateJson"
    params = {
        # 공공데이터의 일반인증키를 온라인URL디코더로 돌려서넣기
        'serviceKey': 'mJlwmW+EQ63NtCKgIG1ZyeTF2mxRv3RLBlKtb7FOF6/E1sVXJ3UTl/99OFJRLCvesPnQ8Nl8IBa4p0fjl6WGLw==',
        'pageNo': '1',
        'numOfRows': '30',
        'startCreateDt': startCreateDt,
        'endCreateDt': endCreateDt
    }

    res = requests.get(url, params=params)
    # print(res.text)
    # print(res.url)

    # xml to dict
    dict_data = xmltodict.parse(res.text)
    # print(dict_data)

    # dict to json
    json_data = json.dumps(dict_data)
    # print(json_data, type(json_data)) #딕셔너리처럼써있는것이 json방식.타입은 str

    # json to dict
    dict_data = json.loads(json_data)
    # print(dict_data, type(dict_data)) #타입 dict가 나와야하는데..왜 str나오지?ㅋㅋㅋ
    # pprint(dict_data['response']['body']['items']['item']) #합계 서울이 마지막으로 나옴.. 뒤집어주려고 아래와같이 실행

    # totalcount로 제대로 데이터가 왔는지 확인하기
    totalcount = dict_data['response']['body']['totalCount']
    #print(type(totalCount)) #토탈카운트가 숫자가 아니라 str이였다
    if totalcount == "0" :
        return False

    # 지역정보를 담은 리스트
    area_data = dict_data['response']['body']['items']['item']
    area_data.reverse()  # 뒤집었기때문에 합계/서울/부산/대구 순으로 나옴
    # print(area_data) #너무 다닥다닥붙어있다 ㅠ 아래와같이 실행

    # for a in area_data:  # 줄별로 확인
    #    print(a)
    return area_data
