import pandas as pd
from rapidfuzz import process, fuzz

# Load and clean the reference dataset ===
ref_df = pd.read_csv('indian cities states.csv', header=1)  # Skip junk header if needed
ref_df.columns = ref_df.columns.str.strip().str.lower()

# Debug: Print columns to confirm structure
print("Cleaned reference columns:", ref_df.columns.tolist())

# Rename and filter reference dataset
ref_df = ref_df.rename(columns={
    'city/town': 'city',
    'state/                                               union territory*': 'state'
})
ref_df = ref_df[['city', 'state']]
ref_df['city'] = ref_df['city'].str.strip().str.lower()
ref_df['state'] = ref_df['state'].str.strip().str.lower()

# Load and clean your working dataset ===
city_df = pd.read_csv('Recommendation_data.csv')
city_df.columns = city_df.columns.str.strip().str.lower()

# Normalize city_name column
city_df['city_name'] = city_df['city_name'].str.strip().str.lower()

# === STEP 3: Manual city aliases ===
city_aliases = {
    'gurgaon': 'gurugram',
    'bombay': 'mumbai',
    'madras': 'chennai',
    'calcutta': 'kolkata',
    'trivandrum': 'thiruvananthapuram',
}

# Extract unique cities and apply aliases
unique_cities = city_df['city_name'].drop_duplicates()
unique_cities = unique_cities.apply(lambda c: city_aliases.get(c, c))

# Fuzzy match cities ===
ref_city_list = ref_df['city'].tolist()
matched_cities = []

for city in unique_cities:
    match, score, _ = process.extractOne(city, ref_city_list, scorer=fuzz.token_sort_ratio)
    if score >= 85:
        state = ref_df.loc[ref_df['city'] == match, 'state'].values[0]
        matched_cities.append({
            'city_name': city,
            'matched_city': match,
            'state': state
        })
    else:
        matched_cities.append({
            'city_name': city,
            'matched_city': None,
            'state': None
        })

# Manual corrections ===
manual_fixes = {
    'mohali':           ('mohali', 'punjab'),
    'gurugram':         ('gurugram', 'haryana'),
    'navi':             ('navi mumbai', 'maharashtra'),
    'delhi':            ('delhi', 'delhi'),
    'mumbai':           ('mumbai', 'maharashtra'),
    'sangli':           ('sangli', 'maharashtra'),
    'panchkula':        ('panchkula', 'haryana'),
    'hyderabad':        ('hyderabad', 'telangana'),
    'howrah':           ('howrah', 'west bengal'),
    'ranga':            ('ranga reddy', 'telangana'),
}

for row in matched_cities:
    if not row['state'] or row['matched_city'] != row['city_name']:
        city_key = row['city_name'].lower()
        if city_key in manual_fixes:
            row['matched_city'], row['state'] = manual_fixes[city_key]

# Create and format output ===
mapped_df = pd.DataFrame(matched_cities)

# Convert names to title case for output
mapped_df['city_name'] = mapped_df['city_name'].str.title()
mapped_df['matched_city'] = mapped_df['matched_city'].str.title()
mapped_df['state'] = mapped_df['state'].str.title()

# Export to CSV
mapped_df.to_csv('Mapped_city.csv', index=False)
print("Mapped_city.csv created successfully!")
