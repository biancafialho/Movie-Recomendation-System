import streamlit as st
import pandas as pd
import seaborn as sns
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split

datam = pd.read_csv('movies.CSV', sep=';', encoding='unicode_escape', decimal=",")
datam=datam.drop(columns=['Timestamp'])
datam.UserId = datam.UserId.astype('category')
datam.ItemId = datam.ItemId.astype('category')
new_df=datam.groupby("ItemId").filter(lambda x:x['Rating'].count() >=50)

movie = pd.read_csv('Itens.csv', sep=';', encoding='unicode_escape', decimal=",")

st.set_page_config(
    page_title="Find a Movie!", page_icon="🎬", layout="wide", initial_sidebar_state="expanded"
)
sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Find a Movie!')

with row0_2:
    st.write('')

row0_2.subheader(
    '10 filmes para você assistir hoje!')

row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown("Aqui você pode sair do tédio! Através de uma análise de mais de mil filmes, encontramos os melhores para você.")

row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
with row2_1:
    default_movie = st.selectbox("Para começar, escolha o filme que você mais gosta: 👇", (
        "Schindler's List (1993)", "Shawshank Redemption, The (1994)", "Casablanca (1942)", "Star Wars (1977)", "Usual Suspects, The (1995)", "To Kill a Mockingbird (1962)", "Silence of the Lambs, The (1991)","Godfather, The (1972)", "Close Shave, A (1995)", "Wrong Trousers, The (1993)"))
user_input = default_movie

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df,reader)
#trainset, testset = train_test_split(data, test_size=0.1,random_state=10)
trainset=data.build_full_trainset()
algo = KNNWithMeans(k=5, sim_options={'user_based': False})
algo.fit(trainset)


def get_movie_data(user_input):
    movie_id = movie[movie['Nome'] == user_input]['Codigo'].values[0]  
    recomendados = algo.get_neighbors(movie_id, 10) # 1 é o filme base, 10 é o número de filmes que serão recomendados

    a=[]
    for i in recomendados:
        a.append(movie.iloc[i]['Nome'])

    return a

x = get_movie_data(user_input)
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown('Você gostaria de assistir:')
for j in x:
    row4_spacer1, row4_1, row4_spacer2 = st.columns((.1, 3.2, .1))
    with row4_1:
        st.markdown(f'🎥{j}')











  