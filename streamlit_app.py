import streamlit as st
import pandas as pd, numpy as np
from sklearn.neighbors import NearestNeighbors

# Load data (make sure these paths are correct)
df = pd.read_csv("model_data.csv")
non_transform_df = pd.read_csv("model_data_pre-transform.csv")
data = df.drop(columns=['player_id', 'Full Name', 'team_name', 'year'])

# Model Initialization
knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
knn.fit(data)

# --- Streamlit UI ---
st.title("Player Similarity App")  # Add a title

# Year selection
years = sorted(df['year'].unique().tolist())
default_year = max(years) if years else None #Default year handling if the list is empty
selected_year = st.selectbox("Select Year", years, index=years.index(default_year) if default_year in years else 0) if years else None

# Player selection (dynamically populated based on selected year)
if selected_year is not None: #Checks if a year has been selected
    players_in_year = df[df['year'] == selected_year]['Full Name'].tolist()

    if players_in_year:
        selected_player = st.selectbox("Select Player", players_in_year)

        try:
            index_input = df[(df['Full Name'] == selected_player) & (df['year'] == selected_year)].index[0]
            similarity(selected_player, selected_year, index_input)

        except IndexError:
            st.error(f"No data found for {selected_player} in {selected_year}. Please select a different player or year.")

    else:
        st.write(f"No players found for the year {selected_year}")
else:
    st.write("No years available in the data.")


# --- Similarity Function ---
def similarity(name_input, year_input, index_input):
    if not name_input or not year_input or index_input is None:
        st.error("Invalid input. Please make sure to select a player and year.")
        return

    target_stats = data.iloc[index_input].values.reshape(1, -1)
    target_stats_df = pd.DataFrame(target_stats, columns=data.columns)

    distances, indices = knn.kneighbors(target_stats_df)

    valid_indices = []
    seen_names = set()

    for i in indices[0]:
        similar_name = df.iloc[i]['Full Name']
        similar_year = df.iloc[i]['year']
        if similar_name != name_input and similar_name not in seen_names and similar_year == year_input: #Added year filter
            valid_indices.append(i)
            seen_names.add(similar_name)
        if len(valid_indices) == 5:
            break

    st.write(f"Target Player's Stats for {name_input} in {year_input}:")
    target_player_info = non_transform_df.iloc[index_input]
    st.write(target_player_info[['Full Name', 'year', *data.columns]])

    if valid_indices:
        similar_player_info = non_transform_df.iloc[valid_indices]
        st.write(f"5 most similar players to {name_input} in {year_input}:")
        st.write(similar_player_info[['Full Name', 'year', *non_transform_df.columns]])
    else:
        st.write(f"No similar players found for {name_input} in {year_input}")
