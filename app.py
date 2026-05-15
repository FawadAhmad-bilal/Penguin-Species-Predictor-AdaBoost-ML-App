import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Penguin Species Predictor", layout="wide", page_icon="🐧")

st.title("🐧 Penguin Species Prediction System")
st.subheader("Advanced Machine Learning Classification using AdaBoost")
st.divider()

@st.cache_resource
def load_model():
    with open('adaboost_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except:
    st.error("⚠️ Model file not found! Please ensure 'adaboost_model.pkl' is in the directory.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Batch Analysis", "ℹ️ About"])

with tab1:
    st.header("Single Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location & Measurements")
        island = st.selectbox("🏝️ Island", ["Torgersen", "Biscoe", "Dream"], help="Penguin's habitat island")
        bill_length = st.slider("📏 Bill Length (mm)", 30.0, 60.0, 39.1, 0.1)
        bill_depth = st.slider("📐 Bill Depth (mm)", 13.0, 22.0, 18.7, 0.1)
    
    with col2:
        st.subheader("⚖️ Physical Attributes")
        flipper_length = st.slider("🦅 Flipper Length (mm)", 170.0, 240.0, 181.0, 1.0)
        body_mass = st.slider("⚖️ Body Mass (g)", 2700.0, 6300.0, 3750.0, 50.0)
        sex = st.radio("⚥ Sex", ["Male", "Female"], horizontal=True)
    
    st.divider()
    
    predict_button = st.button("🚀 Predict Species", type="primary", use_container_width=True)
    
    if predict_button:
        island_map = {"Torgersen": 2, "Biscoe": 0, "Dream": 1}
        sex_map = {"Male": 1, "Female": 0}
        
        input_data = pd.DataFrame({
            'island': [island_map[island]],
            'bill_length_mm': [bill_length],
            'bill_depth_mm': [bill_depth],
            'flipper_length_mm': [flipper_length],
            'body_mass_g': [body_mass],
            'sex': [sex_map[sex]]
        })
        
        prediction = model.predict(input_data)[0]
        # proba = model.predict_proba(input_data)[0]
        
        species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
        predicted_species = species_map[prediction]
        
        st.balloons()
        st.success(f"### 🎯 Predicted Species: **{predicted_species}**")
        
        st.subheader("📊 Prediction Confidence")
        
        col1, col2, col3 = st.columns(3)
        
        st.divider()
        
        with st.expander("📋 Input Summary"):
            summary_df = pd.DataFrame({
                'Feature': ['Island', 'Bill Length', 'Bill Depth', 'Flipper Length', 'Body Mass', 'Sex'],
                'Value': [island, f"{bill_length} mm", f"{bill_depth} mm", f"{flipper_length} mm", f"{body_mass} g", sex]
            })
            st.table(summary_df)

with tab2:
    st.header("📊 Batch Prediction Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file with penguin data", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📂 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info(f"📌 Total Records: {len(df)} | Columns: {len(df.columns)}")
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing predictions..."):
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)
                
                species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
                df['Predicted_Species'] = [species_map[p] for p in predictions]
                # df['Confidence'] = [f"{max(prob)*100:.2f}%" for prob in probabilities]
                
                st.success(f"✅ Successfully predicted {len(df)} records!")
                
                st.subheader("📊 Results")
                st.dataframe(df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    species_counts = df['Predicted_Species'].value_counts()
                    fig_pie = px.pie(
                        values=species_counts.values,
                        names=species_counts.index,
                        title="Species Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = px.bar(
                        x=species_counts.index,
                        y=species_counts.values,
                        labels={'x': 'Species', 'y': 'Count'},
                        title="Species Count",
                        color=species_counts.index,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("👆 Please upload a CSV file to perform batch predictions")

with tab3:
    st.header("ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.write("""
        This application uses **AdaBoost (Adaptive Boosting)** machine learning algorithm 
        to classify penguin species based on physical measurements.
        """)
        
        st.subheader("📊 Model Information")
        st.write("""
        - **Algorithm**: AdaBoost Classifier
        - **Base Estimator**: Decision Tree
        - **Accuracy**: ~98.5%
        - **Species**: Adelie, Chinstrap, Gentoo
        """)
    
    with col2:
        st.subheader("📏 Input Features")
        st.write("""
        1. **Island**: Torgersen, Biscoe, or Dream
        2. **Bill Length**: Measured in millimeters
        3. **Bill Depth**: Measured in millimeters
        4. **Flipper Length**: Measured in millimeters
        5. **Body Mass**: Measured in grams
        6. **Sex**: Male or Female
        """)
        
        st.subheader("👨‍💻 Developer")
        st.write("**Fawad Ahmad Bilal- BSAI Student**")
        st.write("**Roll no F24-3079**")
        st.write("University of Haripur")
    
    st.divider()
    

st.divider()
