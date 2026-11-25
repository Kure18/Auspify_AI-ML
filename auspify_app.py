import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title("🎬 Netflix Content Analytics App")

# ==========================================
# TASK 1: LOAD & CLEAN DATA
# ==========================================
@st.cache_data
def get_data():
    encodings = ['utf-8', 'cp1252', 'latin1']

    for encoding in encodings:
        try:
            df = pd.read_csv('https://drive.google.com/uc?export=download&id=1IzNlvQHIGNdd2KtQkGap6O-vtrKzag6Y', encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode CSV file using supported encodings.")
    
    for col in ['type', 'listed_in', 'title', 'director', 'country', 'rating']:
        df[col] = df[col].fillna('unknown').astype(str)
    return df

df = get_data()

# ==========================================
# TASK 2 & 6: EXPLORATORY DATA ANALYSIS
# ==========================================
st.header("1. Catalog Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Titles", len(df))
col2.metric("Movies %", f"{(df['type'] == 'Movie').mean() * 100:.1f}%")
col3.metric("TV Shows %", f"{(df['type'] == 'TV Show').mean() * 100:.1f}%")

st.plotly_chart(px.histogram(df, x='rating', color='type', title="Content by Rating"))

# ==========================================
# TASK 3: CONTENT RECOMMENDER
# ==========================================
st.header("2. Content Recommender")

@st.cache_resource
def get_similarity(data):
    clean = lambda x: str.lower(x.replace(" ", ""))
    clean_cat = lambda x: " ".join([clean(c) for c in x.split(',')])
    
    meta = (
        data['listed_in'].apply(clean_cat) + " " + data['listed_in'].apply(clean_cat) + " " +
        data['director'].apply(clean) + " " + data['director'].apply(clean) + " " +
        data['country'].apply(clean) + " " + data['release_year'].astype(str) + " " +
        data['title'].str.lower()
    )
    matrix = TfidfVectorizer(stop_words='english').fit_transform(meta)
    return cosine_similarity(matrix, matrix), pd.Series(data.index, index=data['title']).drop_duplicates()

cosine_sim, indices = get_similarity(df)

title_choice = st.selectbox("Select a Title:", sorted(df['title'].unique()))
type_filter = st.radio("Filter Type:", ["All", "Movie", "TV Show"], horizontal=True)

if st.button("Recommend"):
    idx = indices[title_choice]
    if isinstance(idx, pd.Series): 
        idx = idx.iloc[0]
        
    scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    recs = [
        i for i, _ in scores 
        if i != idx and (type_filter == "All" or df.iloc[i]['type'] == type_filter)
    ][:5]
    st.dataframe(df[['title', 'type', 'director', 'listed_in']].iloc[recs], hide_index=True)

# ==========================================
# TASK 4: TIME SERIES FORECAST
# ==========================================
st.header("3. Release Forecast (2021–2025)")

yearly = df.groupby('release_year').size().reset_index(name='count')
yearly['date'] = pd.to_datetime(yearly['release_year'], format='%Y')
train = yearly.set_index('date')['count'].asfreq('YS').fillna(0)['2000':'2020']

forecast = ExponentialSmoothing(train, trend='add', damped_trend=True).fit().forecast(5)

chart_df = pd.DataFrame({
    'Year': train.index.append(forecast.index),
    'Titles': list(train.values) + list(forecast.values.round(0)),
    'Type': ['Historical'] * len(train) + ['Forecast'] * len(forecast)
})
st.plotly_chart(px.line(chart_df, x='Year', y='Titles', color='Type', markers=True))

# ==========================================
# TASK 5: MOVIE VS. TV SHOW CLASSIFIER
# ==========================================
st.header("4. Predict Content Type (ML)")

@st.cache_resource
def get_classifier(data):
    X = data[['title', 'director', 'country', 'rating', 'release_year']]
    y = data['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    prep = ColumnTransformer([
        ('t', TfidfVectorizer(max_features=1000, stop_words='english'), 'title'),
        ('d', TfidfVectorizer(max_features=500), 'director'),
        ('c', OneHotEncoder(handle_unknown='ignore'), ['country', 'rating']),
        ('y', StandardScaler(), ['release_year'])
    ])
    pipe = Pipeline([('prep', prep), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, acc

model, acc = get_classifier(df)
st.write(f"**Model Test Accuracy:** {acc * 100:.2f}%")

c1, c2 = st.columns(2)
in_title = c1.text_input("Title", "Stranger Things")
in_director = c2.text_input("Director", "unknown")
in_country = c1.selectbox("Country", sorted(df['country'].unique()))
in_rating = c2.selectbox("Rating", sorted(df['rating'].unique()))
in_year = st.number_input("Release Year", 1920, 2030, 2022)

if st.button("Predict Type"):
    sample = pd.DataFrame([[in_title, in_director, in_country, in_rating, in_year]], 
                          columns=['title', 'director', 'country', 'rating', 'release_year'])
    st.success(f"Prediction: **{model.predict(sample)[0]}**")
