import pandas as pd 
import re 
from datetime import datetime 
import streamlit as st


# Load city-state mapping
mapped_cities_df = pd.read_csv('Mapped_city.csv')

# Normalize column names to lowercase and strip spaces
mapped_cities_df.columns = mapped_cities_df.columns.str.strip().str.lower()

# Rename to ensure consistent usage
mapped_cities_df.rename(columns={'city_name': 'city'}, inplace=True)

# Normalize data values
mapped_cities_df['city'] = mapped_cities_df['city'].str.lower().str.strip()
mapped_cities_df['state'] = mapped_cities_df['state'].str.lower().str.strip()

# Create city-to-state mapping
city_to_state = dict(zip(mapped_cities_df['city'], mapped_cities_df['state']))

def filter_same_state(df, city):
    city = str(city).strip().lower()
    selected_state = city_to_state.get(city)

    if not selected_state:
        return df, f"City '{city}' not found in mapping. Using all cities."

    df = df.copy()
    df['__clean_city'] = df['City'].astype(str).str.lower().str.strip()
    df['__mapped_state'] = df['__clean_city'].map(city_to_state)

    same_state_df = df[df['__mapped_state'] == selected_state].copy()
    return same_state_df.drop(columns=['__clean_city', '__mapped_state']), None


def extract_first_number(text):
    if pd.isna(text):
        return 0 
    text = str(text)
    match = re.search(r'\d+\.?\d*', text)
    return float(match.group()) if match else 0 

def feature_score(df, binary_columns):
    df[binary_columns] = df[binary_columns].apply(pd.to_numeric, errors='coerce')
    df['Feature_score'] = df[binary_columns].sum(axis=1)
    return df

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # RENAME THE COLUMNS
    df.rename(columns={
        'city_name': 'City',
        'price': 'Price_numeric',
        'mileage': 'Distance_numeric'
    }, inplace=True)

    # SPECIFYING THE COLUMNS THAT I NEED 
    columns_to_keep = [
        'Make', 'Model', 'Variant', 'City', 'Price_numeric', 'Distance_numeric', 'year',
        'Mileage (ARAI)', 'NCAP Rating', 'Airbags', 'Seating Capacity', 'Seat Belt Warning', 'Transmission',
        'Body style', 'Engine', 'Engine Type', 'Turbocharger/Supercharger', 'Steering Type',
        'Anti-Lock Braking System (ABS)', 'Child Safety Lock', 'Air Conditioner',
        'Keyless Start/ Button Start', 'Cruise Control', 'Ventilated Seats', 'USB Compatibility'
    ]
  
    df = df[columns_to_keep]

    # YES/NO COLUMN NORMALIZATION
    yes_no_columns = []
    for col in df.columns:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()
        if set(unique_vals).issuperset({'yes','no'}):
            yes_no_columns.append(col)

    # CLEANING THE VALES IN REQUIRED COLUMNS
    replacement_map = {
        'Turbocharger/Supercharger': ['twin turbo', 'turbocharged'],
        'Anti-Lock Braking System (ABS)': ['optional', 'optional (extra)'],
        'Cruise Control': ['adaptive', 'optional'],
        'USB Compatibility': ['optional']
    }
  
    for col, wrong_values in replacement_map.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().replace({val: 'yes' for val in wrong_values})

    # CLEANING THE AIRCONDITIONER COLUMN
    if 'Air Conditioner' in df.columns:
        df['AC_Raw'] = df['Air Conditioner'].astype(str).str.strip()
        df['Air Conditioner Status'] = df['AC_Raw'].str.extract(r'^(yes|no)', flags=re.IGNORECASE)[0].str.lower()
        df['Air Conditioner Type'] = df['AC_Raw'].str.extract(r'\((.*?)\)')
        df.drop(columns=['Air Conditioner', 'AC_Raw'], inplace=True)

    # CLEANING THE VENTILATED SEATS COLUMN
    df.rename(columns={'Ventilated Seats': 'Ventilated Seats Type'}, inplace=True)
    df['Ventilated Seats Status'] = df['Ventilated Seats Type'].apply(
        lambda x: 'no' if str(x).strip().lower() == 'no' else 'yes'
    )

    # CLEANING THE POWER STEERING COLUMNS
    if 'Steering Type' in df.columns:
        df['Power Steering'] = df['Steering Type'].apply(
            lambda x: 'no' if str(x).strip().lower() == 'manual' else 'yes'
        )

    # CALCULATING THE AGE OF THE CAR  
    current_year = datetime.now().year
    df['Car Age'] = current_year - df['year']

    # EXTRACTING REQUIRED COLUMN FROM NCAP RATING
    df['NCAP Rating Numeric'] = df['NCAP Rating'].apply(extract_first_number)
    df['NCAP Tested'] = df['NCAP Rating Numeric'].apply(lambda x: 'yes' if x > 0 else 'no')

    # PROCESSING THE TRANSMISSION COLUMN
    df['Automatic Transmission'] = df['Transmission'].apply(
        lambda x: 'no' if str(x).strip().lower() == 'manual' else 'yes'
    )

    # PROCESSING THE SEATING CAPACITY COLUMN
    df['Seating Capacity'] = df['Seating Capacity'].astype(str)
    df['Seating Capacity'] = df['Seating Capacity'].str.replace(r'\s*(person|seater|seat)s?\s*', '', regex=True)
    seating_map = {'5 & 6': 6, '7 & 8': 8, '7 & 9': 9}
    df['Seating Capacity'] = df['Seating Capacity'].replace(seating_map)
    df['Seating Capacity'] = df['Seating Capacity'].apply(extract_first_number)

    # FINAL UPDATE OF THE YES/NO COLUMNS
    yes_no_columns += [
        'Air Conditioner Status', 'Ventilated Seats Status', 'Power Steering',
        'NCAP Tested', 'Automatic Transmission'
    ]
    yes_no_columns = list(set(yes_no_columns) & set(df.columns))

    # CONVERT THE YES/NO COLUMNS TO BINARY COLUMNS
    for col in yes_no_columns:
        df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # COMPUTE FEATURE SCORE
    df = feature_score(df, yes_no_columns)

    return df, yes_no_columns

def create_type_column(df):
    def seating_group(seats):
        try:
            seats = int(seats)
        except:
            return 'Unknown'
        if seats <= 4:
            return 'Small'
        elif 5 <= seats <= 6:
            return 'Medium'
        else:
            return 'Large'
    
    df = df.copy()
    df['Seating Group'] = df['Seating Capacity'].apply(seating_group)
    df['Type'] = df['Body style'].str.strip() + ' ' + df['Seating Group']
    return df

def generate_selected_car_info(selected_car):
    info = [
        "\n=== Selected Car ===",
        f"Make: {selected_car['Make']}",
        f"Model: {selected_car['Model']}",
        f"Variant: {selected_car.get('Variant', 'N/A')}",
        f"Price: Rs {int(selected_car['Price_numeric']):,}",
        f"Year: {selected_car.get('year', 'N/A')} (Age: {int(selected_car['Car Age'])} years)",
        f"Distance Driven: {int(selected_car['Distance_numeric']):,} km",
        f"Body Style: {selected_car.get('Body style', 'N/A')}",
        f"Seating Capacity: {selected_car.get('Seating Capacity', 'N/A')}",
        f"Feature Score: {int(selected_car.get('Feature_score', 0))}",
        f"NCAP Rating: {selected_car.get('NCAP Rating Numeric', 'N/A')}",
        f"Transmission: {selected_car.get('Transmission', 'N/A')}",
        f"City: {selected_car.get('City', 'N/A')}",
        "=== End Selected Car ==="
    ]
    return "\n".join(info)

def recommend_cars_by_type(
    df, selected_car,
    price_tolerance_percent=5,
    seating_capacity_filter='gte',
    require_better=True,
    top_n=3
):
    selected_price = selected_car['Price_numeric']
    selected_seating = selected_car['Seating Capacity']
    selected_body = selected_car['Body style']
    selected_feature_score = selected_car.get('Feature_score', 0)

    price_lower = selected_price * (1 - price_tolerance_percent / 100)
    price_upper = selected_price * (1 + price_tolerance_percent / 100)

    filtered_df = df[df['Body style'] == selected_body].copy()

    if seating_capacity_filter == 'gte':
        filtered_df = filtered_df[filtered_df['Seating Capacity'] >= selected_seating]

    filtered_df = filtered_df[
        (filtered_df['Price_numeric'] >= price_lower) & 
        (filtered_df['Price_numeric'] <= price_upper)
    ]

    if filtered_df.empty:
        return None

    if require_better:
        def is_better(row):
            return (
                (row['Feature_score'] > selected_feature_score) or
                (pd.to_numeric(row['Mileage (ARAI)'], errors='coerce') > pd.to_numeric(selected_car['Mileage (ARAI)'], errors='coerce')) or
                (pd.to_numeric(row['NCAP Rating Numeric'], errors='coerce') > pd.to_numeric(selected_car['NCAP Rating Numeric'], errors='coerce')) or
                (row['Automatic Transmission'] > selected_car['Automatic Transmission']) or
                (row['Distance_numeric'] < selected_car['Distance_numeric']) or
                (row['Car Age'] < selected_car['Car Age']) or
                (row['Turbocharger/Supercharger'] > selected_car['Turbocharger/Supercharger']) or
                (row['Anti-Lock Braking System (ABS)'] > selected_car['Anti-Lock Braking System (ABS)']) or
                (row['Child Safety Lock'] > selected_car['Child Safety Lock']) or
                (row['Keyless Start/ Button Start'] > selected_car['Keyless Start/ Button Start']) or
                (row['Cruise Control'] > selected_car['Cruise Control']) or
                (row['USB Compatibility'] > selected_car['USB Compatibility']) or
                (row['Air Conditioner Status'] > selected_car['Air Conditioner Status']) or
                (row['Ventilated Seats Status'] > selected_car['Ventilated Seats Status']) or
                (row['Power Steering'] > selected_car['Power Steering']) or
                (row['NCAP Tested'] > selected_car['NCAP Tested'])
            )

        filtered_df = filtered_df[filtered_df.apply(is_better, axis=1)]

    if filtered_df.empty:
        return None

    sorted_df = filtered_df.sort_values(
        by=['Feature_score', 'Distance_numeric', 'Car Age'],
        ascending=[False, True, True]
    )

    return sorted_df.head(top_n)

def generate_comparison_sentence_by_type(selected_car, recommended_car):
    comparisons = []

    if recommended_car['Feature_score'] > selected_car.get('Feature_score', 0):
        comparisons.append(f"a higher feature score ({int(recommended_car['Feature_score'])} vs {int(selected_car.get('Feature_score', 0))})")
    if recommended_car['NCAP Rating Numeric'] > selected_car['NCAP Rating Numeric']:
        comparisons.append(f"higher NCAP Rating ({recommended_car['NCAP Rating Numeric']} vs {selected_car['NCAP Rating Numeric']})")
    if recommended_car['Seat Belt Warning'] > selected_car['Seat Belt Warning']:
        comparisons.append("a seat belt warning feature")
    if recommended_car['Automatic Transmission'] > selected_car['Automatic Transmission']:
        comparisons.append("an automatic transmission")
    if recommended_car['Distance_numeric'] < selected_car['Distance_numeric']:
        comparisons.append(f"lower driven distance ({int(recommended_car['Distance_numeric'])} km vs {int(selected_car['Distance_numeric'])} km)")
    if recommended_car['Car Age'] < selected_car['Car Age']:
        comparisons.append(f"a newer age ({int(recommended_car['Car Age'])} years vs {int(selected_car['Car Age'])} years)")

    non_numeric_cols = [
        'Make', 'Model', 'Variant', 'City', 'Mileage (ARAI)', 'NCAP Rating', 'Airbags',
        'Transmission', 'Body style', 'Engine', 'Engine Type', 'Steering Type',
        'Ventilated Seats Type', 'Air Conditioner Type'
    ]

    for col in non_numeric_cols:
        selected_val = selected_car.get(col, None)
        recommended_val = recommended_car.get(col, None)
        if selected_val != recommended_val and pd.notna(recommended_val):
            comparisons.append(f"{col.replace('_', ' ').title()}: '{recommended_val}'")

    if not comparisons:
        return "No significant improvements found compared to your selected car."
    
    return "Better because it has " + ", ".join(comparisons) + "."

def recommend_cars_with_type_strategy(df, selected_car):
    df_copy = df.copy()  # Work on a copy to avoid modifying original

    recommendations = []
    output = []

    # Add selected car info manually (no generate_selected_car_info() used)
    output.append(f"Selected Car: {selected_car['Make']} {selected_car['Model']} {selected_car['Variant']} at Rs {int(selected_car['Price_numeric']):,}")
    output.append(f"Feature Score: {int(selected_car.get('Feature_score', 0))}")
    output.append(f"Distance Driven: {int(selected_car.get('Distance_numeric', 0)):,} km")
    output.append(f"Car Age: {int(selected_car.get('Car Age', 0))} years")
    output.append(f"NCAP Rating: {selected_car.get('NCAP Rating Numeric', 'N/A')}")
    output.append("\n=== Recommendations ===")

    # 1. Strict suggestion
    first_suggestion = recommend_cars_by_type(
        df_copy, selected_car, seating_capacity_filter='gte', top_n=1
    )
    if first_suggestion is not None and not first_suggestion.empty:
        car1 = first_suggestion.iloc[0]
        recommendations.append(car1)
        df_copy = df_copy.drop(car1.name)

    # 2. Relaxed suggestion
    second_suggestion = recommend_cars_by_type(
        df_copy, selected_car, seating_capacity_filter='any', top_n=1
    )
    if second_suggestion is not None and not second_suggestion.empty:
        car2 = second_suggestion.iloc[0]
        recommendations.append(car2)
        df_copy = df_copy.drop(car2.name)

    # 3. Fallback suggestion (allow not-better cars)
    third_suggestion = recommend_cars_by_type(
        df_copy, selected_car, seating_capacity_filter='any', require_better=False, top_n=1
    )
    if third_suggestion is not None and not third_suggestion.empty:
        car3 = third_suggestion.iloc[0]
        recommendations.append(car3)
        df_copy = df_copy.drop(car3.name)

    # If still fewer than 3 recommendations, fill remaining slots with closest price cars
    while len(recommendations) < 3 and not df_copy.empty:
        df_copy['price_diff'] = abs(df_copy['Price_numeric'] - selected_car['Price_numeric'])
        next_car = df_copy.sort_values(by='price_diff').iloc[0]
        recommendations.append(next_car)
        df_copy = df_copy.drop(next_car.name)

    # Filter out recommendations with no significant improvements
    filtered_recommendations = []
    for car in recommendations:
        comparison = generate_comparison_sentence_by_type(selected_car, car)
        if "No significant improvements" not in comparison:
            filtered_recommendations.append(car)

    # Build multiline string for recommendations
    for i, car in enumerate(filtered_recommendations, start=1):
        output.append(f"\nRecommendation #{i}: {car['Make']} {car['Model']} at Rs {int(car['Price_numeric']):,}")
        output.append(generate_comparison_sentence_by_type(selected_car, car))

    # If no recommendations left after filtering, add a message
    if not filtered_recommendations:
        output.append("\nNo better alternatives found for your selected car.")

    return "\n".join(output)


def recommend_cars_by_price(df, selected_car, price_tolerance_percent=5, top_n=5):
    output = []
    selected_price = selected_car['Price_numeric']
    selected_feature_score = selected_car.get('Feature_score', 0)

    # Add selected car info at the beginning
    output.append(generate_selected_car_info(selected_car))
    output.append("\n=== Recommendations ===")

    price_lower = selected_price * (1 - price_tolerance_percent / 100)
    price_upper = selected_price * (1 + price_tolerance_percent / 100)

    filtered_df = df[
        (df['Price_numeric'] >= price_lower) &
        (df['Price_numeric'] <= price_upper)
    ].copy()

    if filtered_df.empty:
        return "\n".join(output) + f"\n\nNo cars found within ±{price_tolerance_percent}% of Rs {int(selected_price)}."

    filtered_df = filtered_df.drop_duplicates(subset=['Make', 'Model', 'Variant', 'Price_numeric'])
    filtered_df = filtered_df[filtered_df['Feature_score'] > selected_feature_score]

    if filtered_df.empty:
        return "\n".join(output) + "\n\nNo better cars found in the price range."

    filtered_df = filtered_df.sort_values(
        by=['Feature_score', 'Distance_numeric', 'Car Age'],
        ascending=[False, True, True]
    )

    top_cars = filtered_df.head(top_n)

    for i, (_, car) in enumerate(top_cars.iterrows(), start=1):
        output.append(f"\nRecommendation #{i}: {car['Make']} {car['Model']} at Rs {int(car['Price_numeric']):,}")
        output.append(generate_comparison_sentence_by_type(selected_car, car))

    return "\n".join(output)

def recommend_cars_by_same_model(df, selected_car, price_tolerance_percent=5, top_n=5, yes_no_cols=None):
    output = []
    selected_price = selected_car['Price_numeric']
    selected_feature_score = selected_car.get('Feature_score', 0)

    # Add selected car info at the beginning
    output.append(generate_selected_car_info(selected_car))
    output.append("\n=== Recommendations ===")

    # Price range calculation
    price_lower = selected_price * (1 - price_tolerance_percent / 100)
    price_upper = selected_price * (1 + price_tolerance_percent / 100)

    # Filter for same make and model, without considering the variant
    filtered_df = df[
        (df['Make'] == selected_car['Make']) &  # Same make
        (df['Model'] == selected_car['Model']) &  # Same model, any variant
        (df['Price_numeric'] >= price_lower) & 
        (df['Price_numeric'] <= price_upper) & 
        (df.index != selected_car.name)  # Exclude the selected car itself
    ].copy()

    if filtered_df.empty:
        return "\n".join(output) + "\n\nNo other cars found in the same model within the specified price range."

    # If yes_no_cols are provided, calculate the feature score
    if yes_no_cols:
        filtered_df['Feature_score'] = filtered_df[yes_no_cols].sum(axis=1)
    elif 'Feature_score' not in filtered_df.columns:
        raise ValueError("Feature score column missing and yes_no_cols not provided.")

    # Filter by feature score: only recommend cars with a higher feature score than the selected car
    filtered_df = filtered_df[filtered_df['Feature_score'] > selected_feature_score]

    if filtered_df.empty:
        return "\n".join(output) + "\n\nNo better cars found in the same model within the price range."

    # Sorting by feature score, distance (mileage), and car age
    filtered_df = filtered_df.sort_values(
        by=['Feature_score', 'Distance_numeric', 'Car Age'],
        ascending=[False, True, True]
    )

    # Get the top recommended cars
    top_cars = filtered_df.head(top_n)

    # Loop through the top recommended cars and generate output
    for i, (_, car) in enumerate(top_cars.iterrows(), start=1):
        output.append(f"\nRecommendation #{i}: {car['Make']} {car['Model']} {car['Variant']} at Rs {int(car['Price_numeric']):,}")
        output.append(generate_comparison_sentence_by_type(selected_car, car))

    return "\n".join(output)
