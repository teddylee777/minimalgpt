# minimalgpt

최소한의 코드로 랭체인(langchain) 과 OpenAI GPT 모듈 사용하기!

## 설치

```bash
pip install minimalgpt
```

## 예제코드

OPENAI API KEY 입력

```
import os

os.environ['OPENAI_API_KEY'] = 'OPENAI API KEY 를 입력해 주세요'
```

단답형: ChatModule
```python
from minimalgpt.modules import ChatModule

# 객체 생성
chatbot = ChatModule(
    temperature=0, 
    model_name='gpt-4-1106-preview', 
)

# 질의
answer = chatbot.ask('우리나라의 수도는 뭐야?')
```

대화형: ConversationModule
```python
from minimalgpt.modules import ConversationModule

chatbot = ConversationModule(temperature=0, 
                             model_name='gpt-3.5-turbo')

chatbot.ask('다음을 영어로 번역해 줘: 나는 파이썬 프로그래밍을 정말 사랑해')
```

웹크롤링 + 요약: WebSummarizeModule
```python
from minimalgpt.modules import WebSummarizeModule

# 뉴스기사 요약 봇 생성
summarize_bot = WebSummarizeModule(
    url='https://n.news.naver.com/mnews/article/011/0004249701?sid=101', 
    temperature=0, 
    model_name='gpt-4-1106-preview',
)

# 전체 문서에 대한 지시(instruct) 정의
template = '''{text}

요약의 결과는 다음의 형식으로 작성해줘:
제목: 신문기사의 제목
날짜: 작성일
작성자: 기사의 작성자를 기입
주요내용: 한 줄로 요약된 내용
내용: 주요내용을 불렛포인트 형식으로 작성
'''
# 요약
answer = summarize_bot.ask(template)
print(answer)
```

데이터 분석: PandasModule
```python
from minimalgpt.modules import PandasModule

data_anaylsis = PandasModule(
    df=df, 
    temperature=0, 
    model_name='gpt-4-1106-preview',
)

data_anaylsis.ask('나이의 평균과 표준편차를 구해줘')
```
