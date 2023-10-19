from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


class ChatModule():
    def __init__(self, streaming=True, **model_kwargs):
        # 객체 생성
        if streaming:
            model_kwargs['streaming'] = True
            model_kwargs['callbacks'] = [StreamingStdOutCallbackHandler()]
        self.llm = ChatOpenAI(**model_kwargs)
        
    def ask(self, question):
        return self.llm.predict(question)


class ConversationModule():
    def __init__(self, streaming=True, **model_kwargs):
        # 객체 생성
        if streaming:
            model_kwargs['streaming'] = True
            model_kwargs['callbacks'] = [StreamingStdOutCallbackHandler()]
        llm = ChatOpenAI(**model_kwargs)
        # ConversationChain 객체 생성
        self.conversation = ConversationChain(llm=llm)
        
    def ask(self, question):
        return self.conversation.run(question)
    
    
class WebSummarizeModule():
    def __init__(self, url, streaming=False, **model_kwargs):
        # 웹 문서 크롤링
        self.loader = WebBaseLoader(url)
        
        # 뉴스기사의 본문을 Chunk 단위로 쪼갬
        text_splitter = CharacterTextSplitter(        
            separator="\n\n",
            chunk_size=2000,     # 쪼개는 글자수
            chunk_overlap=100,   # 오버랩 글자수
            length_function=len,
            is_separator_regex=False,
        )
        
        # 웹사이트 내용 크롤링 후 Chunk 단위로 분할
        self.docs = WebBaseLoader(url).load_and_split(text_splitter)
        
        # 객체 생성
        if streaming:
            model_kwargs['streaming'] = True
            model_kwargs['callbacks'] = [StreamingStdOutCallbackHandler()]
            
        self.llm = ChatOpenAI(**model_kwargs)
        
    def ask(self, combine_template):
        # 각 Chunk 단위의 템플릿
        template = '''다음의 내용을 한글로 요약해줘:

        {text}
        '''

        # 템플릿 생성
        prompt = PromptTemplate(template=template, input_variables=['text'])
        combine_prompt = PromptTemplate(template=combine_template, input_variables=['text'])

        # 요약을 도와주는 load_summarize_chain
        chain = load_summarize_chain(self.llm, 
                                     map_prompt=prompt, 
                                     combine_prompt=combine_prompt, 
                                     chain_type="map_reduce", 
                                     verbose=False)
        
        return chain.run(self.docs)



class PandasModule():
    def __init__(self, df, streaming=False, **model_kwargs):
        # 객체 생성
        if streaming:
            model_kwargs['streaming'] = True
            model_kwargs['callbacks'] = [StreamingStdOutCallbackHandler()]
        llm = ChatOpenAI(**model_kwargs)
        # 에이전트 생성
        self.agent = create_pandas_dataframe_agent(
            llm,                                   # 모델 정의
            df,                                    # 데이터프레임
            verbose=True,                          # 추론과정 출력
            agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        
    def ask(self, query):
        return self.agent.run(query)

