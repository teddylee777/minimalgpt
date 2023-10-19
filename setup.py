from setuptools import setup, find_packages

setup(
    name='minimalgpt',
    version='0.0.3',
    description='Simple LangChain Wrapper Package for ease of using ChatGPT',
    author='teddylee777',
    author_email='teddylee777@gmail.com',
    url='https://github.com/teddylee777/minimalgpt',
    install_requires=['openai', 'langchain', 'beautifulsoup4', 'tiktoken'],
    packages=find_packages(exclude=[]),
    keywords=['minimalgpt', 'teddylee777', 'chatgpt', 'langchain', 'pandasai'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)