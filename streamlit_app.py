import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap
import mlflow
import mlflow.sklearn



# Load the clean data
df = pd.read_csv('cleaned_fashion_data.csv')

###################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Features and target
features = ['Gender', 'Category', 'PrimaryColor', 'Material', 'Season', 'NumImages']
target = 'PastTrendScore'

# Prepare data
X = df[features]
y = df[target]

# Preprocessing
categorical_features = ['Gender', 'Category', 'PrimaryColor', 'Material', 'Season']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
###############################

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("📂 Navigation", ["🧠 Business Case", "📊 Trend Insights", "🔮 Predict Trends",  "💡 Feature Importance", "🛠️ Model Tuning"])


if page == "🧠 Business Case":
    st.title("Fashion Trend Predictor 🎯")
    st.write("""
    This app helps retailers predict which fashion trends (styles, colors, materials) will be popular 
    in upcoming seasons. We use machine learning models trained on product features.
    """)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

elif page == "📊 Trend Insights":
    st.title("Trend Insights 📊")

    # Bar Plot: Count of Products per Category
    st.subheader("Product Categories")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Category', order=df['Category'].value_counts().index, ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Category")
    st.pyplot(fig1)

    # Pie Chart: Gender Distribution 
    st.subheader("Gender Distribution")
    gender_counts = df['Gender'].value_counts()

    fig2, ax2 = plt.subplots()
    explode = [0.1 if count < 0.05 * gender_counts.sum() else 0 for count in gender_counts]  # explode tiny slices

    ax2.pie(
        gender_counts,
        labels=None,  # hide labels on the pie
        autopct='%1.1f%%',
        startangle=140,
        explode=explode
    )
    ax2.axis('equal')
    ax2.legend(gender_counts.index, title="Gender", loc="center left", bbox_to_anchor=(1, 0.5))
    st.pyplot(fig2)

    # Box Plot: Price by Season
    st.subheader("Price Distribution by Season")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='Season', y='Price (INR)', ax=ax3)
    ax3.set_ylabel("Price (INR)")
    st.pyplot(fig3)

    # Bar Plot: Average Past Trend Score by Primary Color
    st.subheader("Average Past Trend Score by Color")
    avg_score_by_color = df.groupby('PrimaryColor')['PastTrendScore'].mean().sort_values(ascending=False).head(10)
    fig4, ax4 = plt.subplots()
    sns.barplot(x=avg_score_by_color.values, y=avg_score_by_color.index, ax=ax4)
    ax4.set_xlabel("Average Trend Score")
    ax4.set_ylabel("Primary Color")
    st.pyplot(fig4)

    st.markdown("""
### 🛍️ Product Category Insights:
- The most common product categories are **Other** (4493 items), followed closely by **Tops**.
- Categories like **Accessories** are underrepresented.

### 🧑‍🤝‍🧑 Gender Targeting Insights:
- **Women’s** fashion dominates the dataset (**40.8%**), followed by **Men** (**37.4%**).
- Categories like **Girls**, **Unisex Kids**, and **Boys** together account for less than 5% of listings.

### 💰 Price Distribution Insights:
- Average price per season (in INR):

| Season | Mean | Min | Max |
|--------|------|-----|-----|
| Fall   | 1516 | 153 | 58854 |
| Spring | 1476 | 223 | 34500 |
| Summer | 1444 | 199 | 63090 |
| Winter | 1408 | 229 | 24800 |

- Most products are priced under ₹5,000, but outliers exceed ₹60,000.

### 🎨 Top Trending Colors:
- The highest average past trend scores are associated with colors like:

| Color     | Avg. Trend Score |
|-----------|------------------|
| Platinum  | 3.40 |
| Silver    | 3.37 |
| Purple    | 3.36 |
| Matte     | 3.35 |
| Rose      | 3.33 |
| Magenta   | 3.33 |
| Maroon    | 3.31 |
| Black     | 3.30 |
| Gold      | 3.30 |
| Grey      | 3.28 |
""")



elif page == "🔮 Predict Trends":
    st.title("Predict Trend Score 🔮")

    st.write("Fill in the product features below to estimate how trendy this item might be next season.")

    # Input fields
    gender = st.selectbox("Gender", df['Gender'].unique())
    category = st.selectbox("Category", df['Category'].unique())
    color = st.selectbox("Primary Color", df['PrimaryColor'].dropna().unique())
    material = st.selectbox("Material", df['Material'].unique())
    season = st.selectbox("Season", df['Season'].unique())
    num_images = st.slider("Number of Images", min_value=1, max_value=10, value=5)

    model_choice = st.radio("Choose Model", ["Linear Regression", "Random Forest"])

    if st.button("Predict Trend Score"):
        user_input = pd.DataFrame([{
            'Gender': gender,
            'Category': category,
            'PrimaryColor': color,
            'Material': material,
            'Season': season,
            'NumImages': num_images
    }])

    # Choose model pipeline
    model = lr_model if model_choice == "Linear Regression" else rf_model

    prediction = model.predict(user_input)[0]

    # Show predicted score
    st.success(f"🧠 Predicted Trend Score: **{round(prediction, 2)}** (out of 8)")

    # Emoji label
    if prediction > 6:
        st.success("🌟 Hot Trend!")
    elif prediction > 4:
        st.warning("🔥 Emerging Trend")
    else:
        st.error("🧊 Cold Trend")

    # Business recommendation
    st.subheader("📋 Business Action Plan:")
    if prediction > 6:
        st.markdown("- ✅ Prioritize this item in marketing campaigns.\n- ✅ Stock more units.\n- ✅ Feature in seasonal trend collections.")
    elif prediction > 4:
        st.markdown("- ⚡ Monitor interest and test with limited release.\n- ⚡ Boost visibility through social media or influencers.")
    else:
        st.markdown("- 🧹 Low priority item.\n- 🧹 Avoid large inventory commitments.\n- 🧹 Focus on clearance or bundle deals.")

    st.markdown("""
    ### 📈 How This Helps Businesses

    By using machine learning to predict the trend score of fashion items based on their characteristics (like season, color, and style), brands and retailers can make smarter inventory decisions.  
    For example, if a product has a low predicted trend score, the business might reduce its order quantity or focus less on marketing that item.

    Conversely, high-scoring items can be prioritized in campaigns, stocked in greater volume, or featured in trend collections.  
    This proactive insight allows businesses to reduce overstock, increase sales of in-demand items, and better align with evolving consumer preferences.

    In testing, both our models showed strong accuracy with **Linear Regression slightly outperforming Random Forest**; meaning businesses can trust these predictions to guide trend-sensitive decisions.
    """)



elif page == "💡 Feature Importance":
    st.title("What Drives Trend Predictions? 💡")

    st.write("""
    Understanding why a model makes a prediction is critical for business trust and decision-making. 
    We use SHAP (SHapley Additive exPlanations) to interpret feature importance in our machine learning models.
    """)

    X_sample = X_test.sample(100, random_state=42)

    # Use the trained preprocessor
    preprocessor_fitted = rf_model.named_steps['preprocessor']
    X_encoded = preprocessor_fitted.transform(X_sample)

    # Convert sparse matrix to dense
    X_encoded_dense = X_encoded.toarray()

    # Make DataFrame with feature names
    X_encoded_df = pd.DataFrame(X_encoded_dense, columns=preprocessor_fitted.get_feature_names_out())

    # Get trained regressor
    rf_regressor = rf_model.named_steps['regressor']

    # SHAP explanation
    explainer = shap.TreeExplainer(rf_regressor)
    shap_values = explainer.shap_values(X_encoded_df)

    # Plot
    st.subheader("Feature Importance (Random Forest Model)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_encoded_df, plot_type="bar", show=False)
    st.pyplot(fig)

    st.markdown("""
    ### 🧠 Key Insight:
    - Features like **Season**, **Material**, and **Primary Color** have the strongest impact on a product's predicted trend score.
    - Retailers can use this knowledge to focus on designing products with high-impact features that maximize trend success.
    """)


elif page == "🛠️ Model Tuning":
    st.title("Model Tracking and Tuning 🛠️")

    st.write("""
    Here we compare the performance of our two machine learning models using MLFlow. 
    We track their hyperparameters, metrics, and select the best-performing model.
    """)

    # Generate predictions
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

   
    lr_mse = mean_squared_error(y_test, lr_preds)
    rf_mse = mean_squared_error(y_test, rf_preds)

    lr_rmse = np.sqrt(lr_mse)
    rf_rmse = np.sqrt(rf_mse)

    # Log Linear Regression to MLFlow
    with mlflow.start_run(run_name="Linear Regression"):
        mlflow.sklearn.log_model(lr_model, "LinearRegressionModel")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", lr_rmse)
    mlflow.end_run()

    # Log Random Forest to MLFlow
    with mlflow.start_run(run_name="Random Forest"):
        mlflow.sklearn.log_model(rf_model, "RandomForestModel")
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("rmse", rf_rmse)
    mlflow.end_run()

    # Display performance comparison
    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [round(lr_rmse, 3), round(rf_rmse, 3)],
        "Notes": ["Simple, fast", "Handles non-linearity, more complex"]
    })

    st.subheader("Model Performance Comparison")
    st.dataframe(results)

    best_model = results.loc[results['RMSE'].idxmin(), 'Model']
    st.success(f"✅ Best Model: **{best_model}** based on lowest RMSE.")

st.markdown("""
---
Made by Breanna Richard · DS Project · 2025
""")
