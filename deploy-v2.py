import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import uniform
from geopy.geocoders import Nominatim

# Function to preprocess the data
def preprocess_data(df):
    # Convert latitude and longitude to numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['latitude', 'longitude', 'price', 'mean_income', 'mean_customer'])
    
    # Handle missing zipcodes
    for index, row in df[df['zipcode'].isnull()].iterrows():
        zipcode = get_zipcode(row['latitude'], row['longitude'])
        if zipcode:
            df.at[index, 'zipcode'] = zipcode

    # Log transformation
    df['price_log'] = np.log1p(df['price'])
    df['guests_included_log'] = np.log1p(df['guests_included'])
    df['minimum_nights_log'] = np.log1p(df['minimum_nights'])
    df['bedrooms_log'] = np.log1p(df['bedrooms'].fillna(0))

    # Feature encoding (example: categorizing response rate and acceptance rate)
    df['host_response_rate_kategori'] = pd.cut(df['host_response_rate'], bins=[0, 50, 75, 90, 100], labels=[1, 2, 3, 4])
    df['host_acceptance_rate_kategori'] = pd.cut(df['host_acceptance_rate'], bins=[0, 50, 75, 90, 100], labels=[1, 2, 3, 4])
    df['review_scores_rating_kategori'] = pd.cut(df['review_scores_rating'], bins=[0, 50, 75, 90, 100], labels=[1, 2, 3, 4])

    return df

# Function to get zipcode from latitude and longitude
def get_zipcode(lat, lon):
    geolocator = Nominatim(user_agent="geoapi")
    try:
        location = geolocator.reverse((lat, lon), language='en')
        if location and 'postcode' in location.raw['address']:
            return location.raw['address']['postcode']
    except:
        return None

# Streamlit app
st.title('Airbnb Seattle Mean Income and Customer Prediction')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data")
    st.write(df.head())

    # Preprocess the data
    df = preprocess_data(df)
    st.write("Preprocessed Data")
    st.write(df.head())

    # Feature selection
    features = ['review_scores_rating_kategori', 'host_response_rate_kategori',
                'host_acceptance_rate_kategori', 'bedrooms_log', 'price_log',
                'guests_included_log', 'minimum_nights_log']
    X = df[features]
    y_income = df['mean_income']
    y_customer = df['mean_customer']

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply PCA
    n_components = min(X_scaled.shape[0], X_scaled.shape[1], 5)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Split the data
    X_train_pca, X_test_pca, y_train_pca_income, y_test_pca_income, y_train_pca_customer, y_test_pca_customer = train_test_split(
        X_pca, y_income, y_customer, test_size=0.2, random_state=42)

    # Hyperparameter tuning for SVR
    param_distributions = {
        'kernel': ['linear'],
        'C': uniform(1, 11),
        'epsilon': uniform(0.1, 1),
        'gamma': ['scale', 'auto']
    }

    # RandomizedSearch for mean_income
    svr_income = SVR()
    random_search_income = RandomizedSearchCV(
        svr_income, 
        param_distributions, 
        n_iter=20, 
        scoring='r2', 
        cv=3, 
        n_jobs=-1, 
        random_state=42
    )
    random_search_income.fit(X_train_pca, y_train_pca_income)

    # Best model for mean_income
    best_svr_income = random_search_income.best_estimator_

    # RandomizedSearch for mean_customer
    svr_customer = SVR()
    random_search_customer = RandomizedSearchCV(
        svr_customer, 
        param_distributions, 
        n_iter=20, 
        scoring='r2', 
        cv=3, 
        n_jobs=-1, 
        random_state=42
    )
    random_search_customer.fit(X_train_pca, y_train_pca_customer)

    # Best model for mean_customer
    best_svr_customer = random_search_customer.best_estimator_

    # Predict using the best models
    y_income_pred = best_svr_income.predict(X_pca)
    y_customer_pred = best_svr_customer.predict(X_pca)

    st.write("Mean Predicted Income:", np.mean(y_income_pred))
    st.write("Mean Predicted Customer Count:", np.mean(y_customer_pred))