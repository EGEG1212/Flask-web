import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.parse import quote
import math

def kin_naver_search10P():
    gift_text = ''##추가1
    for page in range(1,11):                            #지식인에서 1~10페이지찾아들어가기
        url = f'{kin_url}?query={search}&page={page}'
        driver.get(url)
        return print(page, end=' ')##몇페이지했나 찍어보기 tqdm비슷ㅋ
        time.sleep(1)

        ul = driver.find_element_by_css_selector('.basic1')
        lis = ul.find_elements_by_tag_name('li')

        ans_href_list = []
        for li in lis:                                  #답변페이지 href찾아내기
            atag = li.find_element_by_tag_name('a')
            ans_href = atag.get_attribute('href')
            ans_href_list.append(ans_href)
        time.sleep(1)

        for ans_href in ans_href_list:                  #각각에 대한 답변카운트찾아내기
            driver.get(ans_href)
            time.sleep(1)
            count = int(driver.find_element_by_css_selector('._answerCount.num').text)
            #print(count)
            for next in range(math.floor((count-1)/5)):     #답변카운트에 맞춰 더보기 클릭
                more = driver.find_element_by_id('nextPageButton')
                more.click()
                time.sleep(1)
                                                        #답변갯수만큼 CSS Selector셀렉트찾아 텍스트꺼내기 > 말뭉치완성
            answers = driver.find_elements_by_class_name('_endContentsText.c-heading-answer__content-user')
            for answer in answers:
                #print('#', sep='', end='')
                gift_text += '\n' + answer.text##추가2
            #print()
            time.sleep(1)