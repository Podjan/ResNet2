import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# Load the saved models
pca = joblib.load('pca.pkl')
scaler = joblib.load('scaler.pkl')
best_svr_income = joblib.load('best_svr_income.pkl')
best_svr_customer = joblib.load('best_svr_customer.pkl')

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


    # Apply PCA
    X_pca = pca.transform(X)

    # Predict using the loaded models
    y_income_pred = best_svr_income.predict(X_pca)
    y_customer_pred = best_svr_customer.predict(X_pca)

    st.write("Mean Predicted Income:", np.mean(y_income_pred))
    st.write("Mean Predicted Customer Count:", np.mean(y_customer_pred))

    st.subheader("Analysis of PCA Components and Feature Importance")
     # PCA components that most influence the target
    feature_importance_income = pd.DataFrame({
        'Feature': [f'PC{i+1}' for i in range(pca.n_components_)],
        'Importance': np.abs(best_svr_income.coef_).flatten()
    }).sort_values(by='Importance', ascending=True)

    feature_importance_customer = pd.DataFrame({
        'Feature': [f'PC{i+1}' for i in range(pca.n_components_)],
        'Importance': np.abs(best_svr_customer.coef_).flatten()
    }).sort_values(by='Importance', ascending=True)

    most_important_pc_income = feature_importance_income.iloc[-1]
    st.write("PCA most influences mean_income:")
    st.write(most_important_pc_income)

    most_important_pc_customer = feature_importance_customer.iloc[-1]
    st.write("PCA most influences mean_customer:")
    st.write(most_important_pc_customer)

    # Loading matrix
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
        index=features
    )

    # The original features that contribute most to an essential PC
    loadings_for_key_pc_income = loadings[most_important_pc_income['Feature']].sort_values(key=abs, ascending=False)
    loadings_for_key_pc_customer = loadings[most_important_pc_customer['Feature']].sort_values(key=abs, ascending=False)

    st.write(loadings_for_key_pc_income)
    st.write(loadings_for_key_pc_customer)

    # Plotting the contributions for mean_income
    plt.figure(figsize=(10, 6))
    plt.barh(loadings_for_key_pc_income.index, loadings_for_key_pc_income.values, color='purple')
    plt.xlabel('Loading Contribution')
    plt.title(f'Original Feature Contribution on {most_important_pc_income['Feature']}')
    st.pyplot(plt)

     # Plotting the contributions for mean_customer
    plt.figure(figsize=(10, 6))
    plt.barh(loadings_for_key_pc_customer.index, loadings_for_key_pc_customer.values, color='orange')
    plt.xlabel('Loading Contribution')
    plt.title(f'Original Feature Contribution on {most_important_pc_customer["Feature"]}')
    st.pyplot(plt)

