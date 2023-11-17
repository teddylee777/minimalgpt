from setuptools import setup, find_packages

setup(
    name='minimalgpt',
    version='0.0.9',
    description='Simple LangChain Wrapper Package for ease of using ChatGPT',
    author='teddylee777',
    author_email='teddylee777@gmail.com',
    url='https://github.com/teddylee777/minimalgpt',
    install_requires=[
        'beautifulsoup4==4.12.2', 
        'tabulate==0.9.0',
        'openai==0.28.1', 
        'langchain==0.0.336', 
        'tiktoken==0.5.1', 
        'langchain-experimental==0.0.41',
        'pandas==1.5.3',
    ],
    packages=find_packages(exclude=[]),
    keywords=['minimalgpt', 'teddylee777', 'chatgpt', 'langchain', 'pandasai'],
    python_requires='>=3.7',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)