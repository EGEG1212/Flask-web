# 시간을 잘 가져오는지 확인
from datetime import date, timedelta
import corona_data

now = date.today()
# print(now)   #2020-12-21로 출력됨 이건 우리가원하는방식이아니랍니다...
# print(now.strftime("%Y%m%d"))  # 연월일 '대소소'(그냥외우기ㅋ) 시분초 '대대대'
# 20201221로출력됨 나이쓰~
now_str = now.strftime("%Y%m%d")
print(now_str)

#오늘날짜로요청
data = corona_data.get_corona_data(20201222, 20201222)  # 일부러 에러일으키려고 내일날짜입력
# print(data)
# 테스트로 돌렸을때 오늘날짜와False가 나와야하는데.. 아래와같은 에러남(totalCount 숫자가 아니라 스트링이였어)

print(data)
# 날짜없으면 어제 날짜로 요청
if not data:
    yesrterday = now - timedelta(days=1)
    yesrterday_str = yesrterday.strftime("%Y%m%d")
    print(yesrterday_str)

    data = corona_data.get_corona_data(yesrterday_str, yesrterday_str)
    print(data)
