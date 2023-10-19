# minimalgpt

최소한의 코드로 랭체인(langchain) 과 OpenAI GPT 모듈 사용하기!

## 설치

```bash
pip install minimalgpt
```

## 예제코드

```python
import os
from minimalgpt.modules import PandasModule

os.environ['OPENAI_API_KEY'] = 'OPENAI API KEY 입력'

# 판다스
bot = PandasModule(df, model_name='gpt-4-0613')
# 질의
bot.ask('남자와 여자의 생존율의 차이는 어떻게 돼?')
```
