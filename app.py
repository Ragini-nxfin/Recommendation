import streamlit as st
import pandas as pd
from car_utils import (
    preprocess_data,
    create_type_column,
    recommend_cars_with_type_strategy,
    recommend_cars_by_price,
    recommend_cars_by_same_model,
    filter_same_state
)
from llm import LLM  # Make sure this is properly imported

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="Car Recommendation App", layout="centered")
st.title("Car Recommendation App")
st.write("Compare your selected car with better options based on feature score, safety, and other key aspects.")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    try:
        df, yes_no_cols = preprocess_data("Recommendation_data.csv")
        df = create_type_column(df)
        return df, yes_no_cols
    except FileNotFoundError:
        st.error("'Recommendation_data.csv' not found in your project folder.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

df, yes_no_cols = load_data()

# ---------- Session State for Recommendations ----------
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'refined_summary' not in st.session_state:
    st.session_state.refined_summary = None

# ---------- UI: Car Selection ----------
if df is not None:
    st.subheader("Select Your Car")

    city = st.selectbox("City", sorted(df['City'].dropna().unique()))
    make_df = df[df['City'] == city]

    make = st.selectbox("Make", sorted(make_df['Make'].dropna().unique()))
    model_df = make_df[make_df['Make'] == make]

    model = st.selectbox("Model", sorted(model_df['Model'].dropna().unique()))
    variant_df = model_df[model_df['Model'] == model]

    variant = st.selectbox("Variant", sorted(variant_df['Variant'].dropna().unique()))
    price_df = variant_df[variant_df['Variant'] == variant]

    price = st.selectbox("Price (₹)", sorted(price_df['Price_numeric'].unique()), format_func=lambda x: f"₹{int(x):,}")
    distance_df = price_df[price_df['Price_numeric'] == price]

    distance = st.selectbox("Distance Driven (km)", sorted(distance_df['Distance_numeric'].unique()), format_func=lambda x: f"{int(x):,} km")
    age_df = distance_df[distance_df['Distance_numeric'] == distance]

    age = st.selectbox("Car Age (years)", sorted(age_df['Car Age'].unique()), format_func=lambda x: f"{int(x)} years")
    match_df = age_df[age_df['Car Age'] == age]

    if match_df.empty:
        st.warning("No exact match found for your selected car.")
        st.stop()

    selected_car = match_df.iloc[0]

    # ---------- UI: Strategy Selection ----------
    st.markdown("---")
    st.subheader("Choose Recommendation Strategy")

    strategy = st.selectbox("Strategy", [
        "Type-Based (Body Style & Seating)",
        "Price-Based (±5% Range)",
        "Same Variant (Better Feature Score)"
    ])

    # ---------- Recommendation Button ----------
    if st.button("Show Recommendations"):
        st.session_state.recommendations = None
        st.session_state.refined_summary = None
        
        st.success(f"Selected Car: {selected_car['Make']} {selected_car['Model']} {selected_car['Variant']}")

        # Filter data to same-state cars
        state_filtered_df, message = filter_same_state(df, selected_car['City'])
        if message:
            st.info(message)

        # ---------- Strategy Execution ----------
        with st.spinner("Generating recommendations..."):
            try:
                if strategy == "Type-Based (Body Style & Seating)":
                    st.markdown("### Recommendations Based on Type")
                    recommendations = recommend_cars_with_type_strategy(state_filtered_df, selected_car)
                
                elif strategy == "Price-Based (±5% Range)":
                    st.markdown("### Recommendations Within Price Range")
                    recommendations = recommend_cars_by_price(state_filtered_df, selected_car)
                
                elif strategy == "Same Model (Better Variant)":
                    st.markdown("### Better Cars in Same Variant")
                    recommendations = recommend_cars_by_same_model(
                        state_filtered_df, selected_car, yes_no_cols=yes_no_cols
                    )
                
                # Store recommendations in session state
                st.session_state.recommendations = recommendations
                st.write(recommendations)
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.stop()

    # Display recommendations if they exist
    if st.session_state.recommendations:
        st.markdown("---")
        
        # Refinement checkbox
        refine_checkbox = st.checkbox("Give refined summary using AI", key="refine_checkbox")
        
        if refine_checkbox:
            if not st.session_state.refined_summary:
                with st.spinner("Generating refined summary..."):
                    try:
                        st.session_state.refined_summary = LLM(st.session_state.recommendations)
                    except Exception as e:
                        st.error(f"Error generating refined summary: {str(e)}")
                        st.stop()
            
            if st.session_state.refined_summary:
                st.markdown("### Refined Comparison")
                st.markdown(st.session_state.refined_summary)

else:
    st.stop()