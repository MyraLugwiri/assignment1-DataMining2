import streamlit as st
# importing necessary libraries for the webscrapping

import pandas as pd


def organising_data(data):
    new_data = data.groupby('cluster')
    for name, group in new_data:
        st.write(f'Cluster {name}:')
        for index, row in group.iterrows():
            st.markdown(f"[{row['link']}]( {row['link']} )")
    return new_data


def main():
    st.title('News Articles')
    st.caption('News Articles Grouped into 10 clusters')
    st.spinner(text='Loading Articles....')

    # reading the data
    data = pd.read_csv('news_articles.csv')
    data = organising_data(data)


if __name__ == '__main__':
    main()
