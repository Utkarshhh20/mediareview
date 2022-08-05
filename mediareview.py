import streamlit as st
import pickle as pkl
import re
import string
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import hydralit_components as hc
from newscatcherapi import NewsCatcherApiClient
import nltk
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import textblob as tb
data = pd.read_csv(r"C:\Users\Utki\Desktop\code\hackathon\archive\fake_or_real_news.csv")

x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)

menu_data = [
    {'icon': "fa fa-search", 'label':"Fake News Detection"},
    {'icon': "fa fa-comments", 'label':"Sentimental Analysis"},
    {'icon': "bi bi-newspaper", 'label':"Our News"}
]
#    {'icon': "bi bi-telephone", 'label':"Contact us"},
over_theme = {'txc_inactive': "#D3D3D3",'menu_background':'#3948A5','txc_active':'white','option_active':'#3948A5'}
dashboard = hc.nav_bar(
menu_definition=menu_data,
override_theme=over_theme,
home_name='The Media Review',
hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
sticky_nav=True, #at the top or not
sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
use_animation=True,
key='NavBar'
)

if dashboard=='The Media Review':
    logo='''
        <style>
        .logo{
            width: 45%;
            margin-top:20px;
            margin-left:-30px;
            margin-bottom:70px;
        }
        </style>
        <body>
        <center><img src='https://iili.io/U2KnUJ.png' alt='logo' class='logo'></img></center> 
        </body>
        '''
    what='''
    <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
    <style>
    .what{
        font-family: 'Montserrat';
        font-size:1.8em;
        color:blue;
        font-weight:600;
        margin-top:0px;
    }
    </style>
    <body>
        <center><p1 class='what'>What is The Media Review?</p1></center>
    </body>
    '''
    whatinfo='''
    <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
    <style>
    .whatinfo{
        font-family: 'Montserrat';
        font-size:1.2em;
        color:;
        font-weight:600;
        margin-top:80px;
    }
    </style>
    <body>
        <center><p1 class='whatinfo'>The Media Review is a web application developed in order to solve a world scale problem of fake news and its wide spread. We aim to provide a platform or a service where you can verify news for free, obtain news as well as get the overall tone of the news through sentimental analysis.</p1></center>
    </body>
    '''
    whatcan='''
    <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
    <style>
    .whatcan{
        font-family: 'Montserrat';
        font-size:1.8em;
        color:blue;
        font-weight:600;
        margin-top:;
    }
    </style>
    <body>
        <center><p1 class='whatcan'>What can I do with The Media Review?</p1></center>
    </body>
    '''
    tech='''
        <style>
        .taimg {
        float: center;
        z-index: 1;
        width: 400px;
        position: relative;
        border-radius: 5%;
        margin-left: 10px;
        }
        </style>
        <body>
        <img src="https://elements-cover-images-0.imgix.net/63e7a835-94a0-48c9-9668-a18bdb38c405?auto=compress%2Cformat&fit=max&w=1370&s=02ee59c23208ebb653e77bc5a4490825" alt="House" class='taimg'></img>
        </body>'''
    techtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .techtxt {
            font-family: 'Montserrat';
            font-size: 25px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='techtxt'> FAKE NEWS DETECTION </p1> </center>
        </body>
        '''
    techsubtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .techsubtxt {
            font-family: 'Montserrat';
            font-size: 15px;
            margin-top:20px;
            font-weight: 600;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='techsubtxt'> Ever wondered whether the news you find online is fake? 
Well, say no more. Introducing, our news reviewer find the credibility of the input information almost in the blink of an eye. Its sources are connected to some of the world’s most trusted news channels and websites. Fake or not? we got you covered…</p1> </center>
        </body>
        '''
    fundament='''
    <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        .faimg {
        float: center;
        z-index: 1;
        width: 400px;
        position: relative;
        border-radius: 5%;
        margin-left: 10px;
        }
        </style>
        <body>
        <img src="https://elements-cover-images-0.imgix.net/a7998dc3-08ad-4035-92d9-b581899e1bd7?auto=compress%2Cformat&fit=max&w=1370&s=e53e18cc42cb19b21c7fd333492cf370" alt="House" class='faimg'></img>
        </body>'''
    fundtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .fundtxt {
            font-family: 'Montserrat';
            font-size: 25px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='fundtxt'> SENTIMENTAL ANALYSIS </p1> </center>
        </body>
        '''
    fundsubtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .fundsubtxt {
            font-family: 'Montserrat';
            font-size: 15px;
            margin-top:20px;
            font-weight: 600;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='fundsubtxt'> Ever noticed how news can morally have an impact on our communities?
With our sentiment analyser one can interpret how a particular news can affect our society. Now, one can always verify if their news will have a positive impact to others before spreading the word with our sentimental analyzer. </p1> </center>
        </body>
        '''
    backt='''
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        .btimg {
        float: center;
        z-index: 1;
        width: 400px;
        position: relative;
        border-radius: 5%;
        }
        </style>
        <body>
        <center><img src="https://elements-cover-images-0.imgix.net/ed6bfb40-0f4a-4441-bce9-345d1bc76700?auto=compress%2Cformat&fit=max&w=1370&s=41c39308ef703798f58781eb09bf208c" alt="House" class='btimg'></img></center>
        <p1 class>
        </body>'''
    bttxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .techtxt {
            font-family: 'Montserrat';
            font-size: 25px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='techtxt'> OUR NEWS </p1> </center>
        </body>
        '''
    btsubtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .btsubtxt {
            font-family: 'Montserrat';
            font-size: 15px;
            margin-top:20px;
            font-weight: 600;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='btsubtxt'> Want to find some of the most credible news from a single word ?
Enter a specific key word and you will be exposed to a vast multitude of credible news regarding the topic of choice. There’s no need to doubt our reputation since our sources are the most trusted so avail the benefits at the earliest.  </p1> </center>
        </body>
        '''
    st.markdown(logo, unsafe_allow_html=True)
    st.markdown(what, unsafe_allow_html=True)
    st.write('')
    blank1,text,blank2=st.columns([0.1,1,0.1])
    st.write('')
    st.write('______________________________________')
    with blank1:
        st.write('')
    with text:
        st.markdown(whatinfo, unsafe_allow_html=True)
    with blank2:
        st.write('')
    st.markdown(whatcan, unsafe_allow_html=True)

    blank1,technical,fundamental,backtest,blank2=st.columns([0.05,1,1,1,0.05])
    with blank1:
        st.write(" ")
    with technical:
        st.markdown(tech, unsafe_allow_html=True)
        st.markdown(techtxt, unsafe_allow_html=True)
        st.write('')
        st.markdown(techsubtxt, unsafe_allow_html=True)
        st.write('____________________')
    with fundamental:
        st.markdown(fundament, unsafe_allow_html=True)
        st.markdown(fundtxt, unsafe_allow_html=True)
        st.write('')
        st.markdown(fundsubtxt, unsafe_allow_html=True)
        st.write('____________________')
    with backtest:
        st.markdown(backt, unsafe_allow_html=True)
        st.markdown(bttxt, unsafe_allow_html=True)
        st.write('')
        st.markdown(btsubtxt, unsafe_allow_html=True)
        st.write('____________________')
    with blank2:
            st.write(" ")
    st.write(' ')
    st.write(' ')
if dashboard=='Sentimental Analysis':
    vader = SentimentIntensityAnalyzer()
    API_KEY = 'fHwC23n7bq5bjthrJE-XSdda1e2mSYsKCy33Ez_S1BQ'
    newscatcherapi = NewsCatcherApiClient(x_api_key=API_KEY)
    header,find=st.columns([1,1])
    with header:
        st.write(' ')
        st.title('Please enter your search 🔍')
    with find:
        search=st.text_input('', value='Apple')
    st.write(' ')
    st.write(' ')
    news_articles = newscatcherapi.get_search(q=search)
    #st.write(news_articles)
    article=news_articles['articles']
    for j in article:
        if j['language']=='en':
            title=j['title']
            author=j['author']
            twitter=j['twitter_account']
            media=j['media']
            rights=j['rights']
            date=j['published_date']
            links=j['link']
            summary=j['summary']
            excerpt=j['excerpt']
            scores = vader.polarity_scores(j['summary'])
            x=tb.TextBlob(j['summary'])
            scorestb=x.sentiment.polarity
            st.title(title)
            st.subheader(author)
            st.write(date)
            st.subheader(excerpt)
            st.write(summary)
            st.write(links)
            st.write('Sentimental Score (Vader Analysis): ', scores)     
            st.write('Sentimental Score (TextBlob Analysis): ', scorestb)
            st.write(rights)
            st.image(media)
            st.write('____________________')
            st.write(' ')
    
if dashboard=='Fake News Detection':
    st.title("Fake News Detection System")
    def fakenewsdetection():
        user = st.text_area("Enter Any News Headline: ", height=300)
        if len(user) < 1:
            st.write("  ")
        else:
            sample = user
            data = cv.transform([sample]).toarray()
            a = model.predict(data)
            for i in a:
                prediction='There is a high chance of the news being '+i
            st.header(prediction)
    fakenewsdetection()

if dashboard=='Our News':
    API_KEY = 'fHwC23n7bq5bjthrJE-XSdda1e2mSYsKCy33Ez_S1BQ'
    newscatcherapi = NewsCatcherApiClient(x_api_key=API_KEY)
    header,find=st.columns([1,1])
    with header:
        st.write(' ')
        st.title('Please enter your search 🔍')
    with find:
        search=st.text_input('', value='Apple')
    st.write(' ')
    st.write(' ')
    news_articles = newscatcherapi.get_search(q=search)
    #st.write(news_articles)
    article=news_articles['articles']
    for j in article:
        if j['language']=='en':
            title=j['title']
            author=j['author']
            twitter=j['twitter_account']
            media=j['media']
            rights=j['rights']
            date=j['published_date']
            links=j['link']
            summary=j['summary']
            excerpt=j['excerpt']
            st.title(title)
            st.subheader(author)
            st.write(date)
            st.subheader(excerpt)
            st.write(summary)
            st.write(links)
            st.write(rights)
            st.image(media)
            st.write('____________________')
            st.write(' ')
