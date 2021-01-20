"# Python-flask-web-2020" 

### Flask는 웹 어플리케이션 개발을 위해 파이썬으로 구동되는 프레임워크이다.  
Flask 프레임워크: https://palletsprojects.com/p/flask/  
Pocco라는 Python 커뮤니티를 이끄는 Armin Ronacher라는 사람이 개발.     
Flask는 Werkzeug WSGI toolkit과 Jinja2 template engine위에 동작한다.  
 물론 파이썬으로 웹 개발을 할 때 많이 쓰이는 프레임워크로 가장 유명한 것은 Django(장고).  
 웹에 대한 이해가 풍부한 사람은 복잡하고 기능이 많은 장고를 잘 활용할 수 있을 것이다.  
flask는 Django프레임워크보다 가볍고, 작은 서버부터 스케일 큰 서버를 만들수 있다.  
Flask는 마이크로 프레임워크이다.  
여기서 마이크로란 핵심기능을 간결하게 유지하되, 차후 어플리케이션을 확장이 가능하게 한다는 것을 뜻한다.     
Flask는 DB를 기본적으로 설정하지 않고 또한 Template engine을 변경하는 것도 쉽다;;.  
 오히려 쉽게 확장할 수 있도록 설계되어 있기 때문에  
 개발을 하는 입장에서는 본인이 필요한 도구와 라이브러리를 자유롭게 선택해서 적용 가능하다는 게 가장 큰 장점이 아닐까 싶다.   
 (데이터베이스를 통합하거나 계정 인증 등을 포함하는 복잡한 앱 뿐만 아니라 단순한 정적 웹 사이트를 만드는 데도 유용하다.)   feat.아무튼워라밸


### Jinja2란
Jinja2(이하 Jinja)는 Python 웹 프레임워크인 Flask에 내장되어 있는 Template 엔진이다.   
동적인 웹 페이지들을 제공하기 위해 특정 데이터를 템플릿과 결합시킨다.
Jinja는 JSP의 문법이나 ES6의 template string과 비슷한 문법을 가지고 있다.  
(이 템플릿엔진을 사용해서 애플리케이션 내 변수와 반복문, 조건문 등을 포함하는 HTML 파일을 렌더링할 수 있다. )  

Jinja 문법은 간단히 아래와 같다.  
> {{ ... }} : 변수나 표현식  
> {% ... %} : if나 for같은 제어문  
> {# ... #} : 주석  
Jinja의 자세한 문법참조 https://jinja.palletsprojects.com/en/2.10.x/   

### Werkzeug(베ㄹㅋ조잌)
Werkzeug는 요청, 응답 객체 그리고 다른 utility 함수를 구현하는 WSGI 툴킷이다.  
WSGI위에 웹 프레임워크가 동작하게 한다.  

### WSGI
WSGI는 Web Server Gateway Interface(WSGI)의 약자로  
Python 웹 어플리케이션의 개발을 위한 인터페이스 표준이다.  
또한 WSGI는 웹 서버와 웹 어플리케이션 간의 데이터 교환을 위한 인터페이스의 명세라고도 할 수 있다.    feat.새로비
"# Flask-web" 
