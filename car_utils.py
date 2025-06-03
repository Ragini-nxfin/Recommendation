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
        st.write(f"City '{city}' not found in mapping. Using all cities.")
        return df

    df = df.copy()
    df['__clean_city'] = df['City'].astype(str).str.lower().str.strip()
    df['__mapped_state'] = df['__clean_city'].map(city_to_state)

    same_state_df = df[df['__mapped_state'] == selected_state].copy()
    return same_state_df.drop(columns=['__clean_city', '__mapped_state'])


def extract_first_number(text):
  if pd.isna(text):
    return 0 
  text = str(text)
  match = re.search(r'\d+\.?\d*', text)
  return float(match.group()) if match else 0 

def feature_score(df, binary_columns):
  df[binary_columns] = df[binary_columns].apply(pd.to_numeric, errors = 'coerce')
  df['Feature_score'] = df[binary_columns].sum(axis=1)
  return df

def preprocess_data(file_path):
  df = pd.read_csv(file_path)

# RENAME THE COLUMNS

  df.rename(columns={
      'city_name': 'City',
      'price': 'Price_numeric',
      'mileage': 'Distance_numeric'
  }, inplace = True)


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

  return df

def create_type_column(df):
    """
    Creates a 'Type' column by categorizing seating capacity as Small, Medium, or Large, 
    and combining it with the body style.
    """
    def seating_group(seats):
        try:
            seats = int(seats)
        except:
            return 'Unknown'  # If seating can't be converted to integer, return 'Unknown'
        if seats <= 4:
            return 'Small'
        elif 5 <= seats <= 6:
            return 'Medium'
        else:
            return 'Large'
    
    df = df.copy()
    df['Seating Group'] = df['Seating Capacity'].apply(seating_group)  # Apply the seating_group function to each row
    df['Type'] = df['Body style'].str.strip() + ' ' + df['Seating Group']  # Create 'Type' by combining Body style and Seating Group
    return df

def recommend_cars_by_type(
    df, selected_car,
    price_tolerance_percent=5,
    seating_capacity_filter='gte',  # 'gte' for strict, 'any' for relaxed
    require_better=True,
    top_n=1  # Number of cars to return
):
    # Extracting necessary values from the selected car
    selected_price = selected_car['Price_numeric']
    selected_seating = selected_car['Seating Capacity']
    selected_body = selected_car['Body style']
    selected_feature_score = selected_car.get('Feature_score', 0)

    # Calculating lower and upper bounds of the price range based on the price tolerance
    price_lower = selected_price * (1 - price_tolerance_percent / 100)
    price_upper = selected_price * (1 + price_tolerance_percent / 100)

    # Filtering the dataframe to include only cars with the same body style as the selected car
    filtered_df = df[df['Body style'] == selected_body].copy()

    # Apply seating capacity filter
    if seating_capacity_filter == 'gte':  # Strict filter: only cars with seating >= selected_seating
        filtered_df = filtered_df[filtered_df['Seating Capacity'] >= selected_seating]
    elif seating_capacity_filter == 'any':  # Relaxed filter: no restriction on seating capacity
        pass

    # Filter by price range ±5%
    filtered_df = filtered_df[
        (filtered_df['Price_numeric'] >= price_lower) & 
        (filtered_df['Price_numeric'] <= price_upper)
    ]

    # If no cars meet the price range, return None
    if filtered_df.empty:
        return None

    # Apply the "better" filter, if required
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


        # Filter the dataframe to include only "better" cars
        filtered_df = filtered_df[filtered_df.apply(is_better, axis=1)]

    # If no cars meet the "better" filter, return None
    if filtered_df.empty:
        return None

    # Sort the filtered cars by Feature Score (desc), Distance (asc), Car Age (asc)
    sorted_df = filtered_df.sort_values(
        by=['Feature_score', 'Distance_numeric', 'Car Age'],
        ascending=[False, True, True]
    )

    # Return the top N cars
    return sorted_df.head(top_n)


def generate_comparison_sentence_by_type(selected_car, recommended_car):
    comparisons = []

    # Compare the selected car with the recommended car
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
        # Add only if different and recommended_val is valid
        if selected_val != recommended_val and pd.notna(recommended_val):
            comparisons.append(f"{col.replace('_', ' ').title()}: '{recommended_val}'")


    # Return a message based on the comparison
    if not comparisons:
        return "No significant improvements found compared to your selected car."
    
    return "Better because it has " + ", ".join(comparisons) + "."


def recommend_cars_with_type_strategy(df, selected_car):
    df_copy = df.copy()  # Work on a copy to avoid modifying original

    # 1. Strict suggestion
    first_suggestion = recommend_cars_by_type(
        df_copy, selected_car, seating_capacity_filter='gte', top_n=1
    )
    if first_suggestion is not None and not first_suggestion.empty:
        car1 = first_suggestion.iloc[0]
        st.write("First suggestion (strict):")
        st.write(f" - {car1['Make']} {car1['Model']} at Rs {int(car1['Price_numeric'])}")
        st.write(generate_comparison_sentence_by_type(selected_car, car1))

        # Remove this car from dataframe for next suggestions
        df_copy = df_copy.drop(car1.name)
    else:
        car1 = None
        st.write("No better cars found for the first suggestion.")

    # 2. Relaxed suggestion
    second_suggestion = recommend_cars_by_type(
        df_copy, selected_car, seating_capacity_filter='any', top_n=1
    )
    if second_suggestion is not None and not second_suggestion.empty:
        car2 = second_suggestion.iloc[0]
        st.write("\nSecond suggestion (relaxed):")
        st.write(f" - {car2['Make']} {car2['Model']} at Rs {int(car2['Price_numeric'])}")
        st.write(generate_comparison_sentence_by_type(selected_car, car2))

        # Remove this car too for fallback
        df_copy = df_copy.drop(car2.name)
    else:
        car2 = None
        st.write("No better cars found for the second suggestion.")

    # 3. Fallback suggestion
    # For fallback, do not require better
    third_suggestion = recommend_cars_by_type(
        df_copy, selected_car, seating_capacity_filter='any', require_better=False, top_n=1
    )
    if third_suggestion is not None and not third_suggestion.empty:
        car3 = third_suggestion.iloc[0]
        st.write("\nThird suggestion (fallback):")
        st.write(f" - {car3['Make']} {car3['Model']} at Rs {int(car3['Price_numeric'])}")
    else:
        car3 = None
        st.write("No better cars found for the fallback suggestion.")

def recommend_cars_by_price(
    df, selected_car, price_tolerance_percent=5, top_n=3
):
    selected_price = selected_car['Price_numeric']
    selected_feature_score = selected_car.get('Feature_score', 0)

    price_lower = selected_price * (1 - price_tolerance_percent / 100)
    price_upper = selected_price * (1 + price_tolerance_percent / 100)

    filtered_df = df[
        (df['Price_numeric'] >= price_lower) &
        (df['Price_numeric'] <= price_upper)
    ].copy()

    if filtered_df.empty:
        st.write(f"No cars found within ±{price_tolerance_percent}% of Rs {int(selected_price)}.")
        return None

    filtered_df = filtered_df.drop_duplicates(subset=['Make', 'Model', 'Variant', 'Price_numeric'])
    filtered_df = filtered_df[filtered_df['Feature_score'] > selected_feature_score]

    if filtered_df.empty:
        st.write("No better cars found in the price range.")
        return None

    filtered_df = filtered_df.sort_values(
        by=['Feature_score', 'Distance_numeric', 'Car Age'],
        ascending=[False, True, True]
    )

    top_cars = filtered_df.head(top_n)

    # Columns groups (adjust as needed)
    numeric_cols = ['Price_numeric', 'Feature_score', 'Distance_numeric', 'Car Age', 'NCAP Rating Numeric', 'Seating Capacity']
    binary_cols = [
        'Automatic Transmission', 'Turbocharger/Supercharger', 'Anti-Lock Braking System (ABS)',
        'Child Safety Lock', 'Seat Belt Warning', 'Keyless Start/ Button Start', 'Cruise Control',
        'USB Compatibility', 'Air Conditioner Status', 'Ventilated Seats Status', 'Power Steering',
        'NCAP Tested'
    ]
    non_numeric_cols = [
        'Make', 'Model', 'Variant', 'City', 'Mileage (ARAI)', 'NCAP Rating', 'Airbags',
        'Transmission', 'Body style', 'Engine', 'Engine Type', 'Steering Type',
        'Ventilated Seats Type', 'Air Conditioner Type'
    ]

    for i, (_, car) in enumerate(top_cars.iterrows(), start=1):
        st.write(f"\nRecommendation #{i}: {car['Make']} {car['Model']} at Rs {int(car['Price_numeric']):,}")

        sentences = []

        # Price comparison
        price_diff = car['Price_numeric'] - selected_car['Price_numeric']
        if abs(price_diff) > 0:
            direction = "lower" if price_diff < 0 else "higher"
            sentences.append(f"This car is priced slightly {direction} than your selected car (Rs {int(car['Price_numeric']):,} vs Rs {int(selected_car['Price_numeric']):,}).")

        # Feature score
        if car['Feature_score'] > selected_car.get('Feature_score', 0):
            sentences.append(f"It offers a higher feature score of {int(car['Feature_score'])} compared to your car's {int(selected_car.get('Feature_score', 0))}.")

        # Distance driven
        dist_diff = car['Distance_numeric'] - selected_car['Distance_numeric']
        if dist_diff != 0:
            direction = "less" if dist_diff < 0 else "more"
            sentences.append(f"It has been driven {abs(int(dist_diff)):,} km {direction} than your car ({int(car['Distance_numeric']):,} km vs {int(selected_car['Distance_numeric']):,} km).")

        # Car age
        age_diff = car['Car Age'] - selected_car['Car Age']
        if age_diff != 0:
            direction = "newer" if age_diff < 0 else "older"
            sentences.append(f"It is {direction} ({int(car['Car Age'])} years vs {int(selected_car['Car Age'])} years).")

        # NCAP rating
        ncap_diff = car['NCAP Rating Numeric'] - selected_car['NCAP Rating Numeric']
        if ncap_diff > 0:
            sentences.append(f"Comes with a better safety rating ({car['NCAP Rating Numeric']} stars vs {selected_car['NCAP Rating Numeric']} stars).")

        # Binary features (only those that the recommended car has and selected does not)
        for col in binary_cols:
            if car.get(col, 0) == 1 and selected_car.get(col, 0) == 0:
                friendly_name = col.replace('_', ' ').lower()
                sentences.append(f"It has {friendly_name} which your car lacks.")

        # Non-numeric features that differ (only show a few important ones for clarity)
        important_non_numeric = ['Body style', 'Transmission', 'Engine Type', 'Air Conditioner Type']
        for col in important_non_numeric:
            car_val = car.get(col, None)
            sel_val = selected_car.get(col, None)
            if pd.notna(car_val) and car_val != sel_val:
                sentences.append(f"The {col.replace('_',' ').lower()} differs: recommended has '{car_val}', yours has '{sel_val}'.")

        # Print detailed description per car
        if sentences:
            st.write("Key differences compared to your selected car:")
            for s in sentences:
                st.write(f" - {s}")
        else:
            st.write("No significant differences found compared to your selected car.")

def recommend_cars_by_same_variant(
    df, selected_car, price_tolerance_percent=5, top_n=3, yes_no_cols=None
):
    selected_price = selected_car['Price_numeric']
    selected_feature_score = selected_car.get('Feature_score', 0)

    # Price range boundaries
    price_lower = selected_price * (1 - price_tolerance_percent / 100)
    price_upper = selected_price * (1 + price_tolerance_percent / 100)

    # Filter cars with same Make, Model, Variant within price range excluding the selected car itself
    filtered_df = df[
        (df['Make'] == selected_car['Make']) &
        (df['Model'] == selected_car['Model']) &
        (df['Variant'] == selected_car['Variant']) &
        (df['Price_numeric'] >= price_lower) &
        (df['Price_numeric'] <= price_upper) &
        (df.index != selected_car.name)
    ].copy()

    if filtered_df.empty:
        st.write("No other cars found in the same variant within the specified price range.")
        return None

    # Calculate feature score if yes_no_cols provided
    if yes_no_cols:
        filtered_df['Feature_score'] = filtered_df[yes_no_cols].sum(axis=1)
    elif 'Feature_score' not in filtered_df.columns:
        raise ValueError("Feature score column missing and yes_no_cols not provided.")

    # Keep only cars with strictly better feature score
    filtered_df = filtered_df[filtered_df['Feature_score'] > selected_feature_score]

    if filtered_df.empty:
        st.write("No better cars found in the same variant within the price range.")
        return None

    # Sort by feature score desc, then distance asc, car age asc
    filtered_df = filtered_df.sort_values(
        by=['Feature_score', 'Distance_numeric', 'Car Age'],
        ascending=[False, True, True]
    )

    top_cars = filtered_df.head(top_n)

    # Columns for comparisons
    numeric_cols = ['Price_numeric', 'Feature_score', 'Distance_numeric', 'Car Age', 'NCAP Rating Numeric', 'Seating Capacity']
    binary_cols = yes_no_cols if yes_no_cols else []
    non_numeric_cols = [
        'Body style', 'Transmission', 'Engine Type', 'Air Conditioner Type',
        'City', 'Mileage (ARAI)', 'NCAP Rating', 'Airbags'
    ]

    for i, (_, car) in enumerate(top_cars.iterrows(), start=1):
        st.write(f"\nRecommendation #{i}: {car['Make']} {car['Model']} {car['Variant']} at Rs {int(car['Price_numeric']):,}")
        sentences = []

        # Numeric differences
        for col in numeric_cols:
            sel_val = selected_car.get(col, None)
            rec_val = car.get(col, None)
            if sel_val is not None and rec_val is not None and rec_val != sel_val:
                if col == 'Price_numeric':
                    direction = "lower" if rec_val < sel_val else "higher"
                    sentences.append(f"Priced slightly {direction} (Rs {int(rec_val):,} vs Rs {int(sel_val):,})")
                elif col == 'Feature_score':
                    sentences.append(f"Higher feature score ({int(rec_val)} vs {int(sel_val)})")
                elif col == 'Distance_numeric':
                    direction = "less" if rec_val < sel_val else "more"
                    sentences.append(f"Driven {direction} ({int(rec_val):,} km vs {int(sel_val):,} km)")
                elif col == 'Car Age':
                    direction = "newer" if rec_val < sel_val else "older"
                    sentences.append(f"{direction.capitalize()} ({int(rec_val)} years vs {int(sel_val)} years)")
                elif col == 'NCAP Rating Numeric':
                    sentences.append(f"Better safety rating ({rec_val} stars vs {sel_val} stars)")
                else:
                    sentences.append(f"{col.replace('_', ' ').title()} differs ({rec_val} vs {sel_val})")

        # Binary features present in recommended car but not in selected
        for col in binary_cols:
            sel_val = selected_car.get(col, 0)
            rec_val = car.get(col, 0)
            if rec_val > sel_val:
                sentences.append(f"Has {col.replace('_', ' ').lower()} which your car lacks")

        # Non-numeric differences
        for col in non_numeric_cols:
            sel_val = selected_car.get(col, None)
            rec_val = car.get(col, None)
            if pd.notna(rec_val) and rec_val != sel_val:
                sentences.append(f"{col.replace('_',' ').title()} differs: recommended has '{rec_val}', yours has '{sel_val}'")

        if sentences:
            st.write("Key differences compared to your selected car:")
            for s in sentences:
                st.write(f" - {s}")
        else:
            st.write("No significant differences found compared to your selected car.")

    return top_cars










  
  
