import streamlit as st
import pandas as pd, numpy as np
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("model_data.csv")
non_transform_df = pd.read_csv("model_data_pre-transform.csv")
data = df.drop(columns=['player_id', 'Full Name', 'team_name', 'year'])
# Model Initialisation

knn = NearestNeighbors(n_neighbors=20, metric='euclidean')  # Use more neighbors to filter duplicates

# Fit KNN model to the data
knn.fit(data)

def user_input():
    # Year selection
    year = st.selectbox("Select a Year", [2020, 2021, 2022, 2023, 2024])  # Adjust range as needed
    
    # Player selection based on year
    name_input = st.text_input("Start typing a player name:")

    # Filter players by name
    filtered_players = df['Full Name'][df['Full Name'].str.contains(name_input, case=False, na=False)].unique().tolist() if name_input else []
    
    player_selection = None
    if filtered_players:
        player_selection = st.selectbox("Select a player", filtered_players)
        st.write(f"You selected: {player_selection}")
    else:
        st.write("No matching players found")

    # Validity check: If no valid input, return None
    if st.button("Find Similar Players"):
        if player_selection not in df['Full Name'].values or year not in df['year'].values:
            st.error("Invalid player or year selected. Please try again.")
        else:
            index_input = df.index.get_loc(df[(df['Full Name'] == player_selection) & (df['year'] == year)].index[0])
            return player_selection, year, index_input

    return None, None, None

# Function to compute similarities
def similarity(name_input, year_input, index_input):
    if not name_input or not year_input or index_input is None:
        st.error("Invalid input. Please make sure to select a player and year.")
        return
    
    target_stats = data.iloc[index_input].values.reshape(1, -1)
    target_stats_df = pd.DataFrame(target_stats, columns=data.columns)

    # Get nearest neighbors based on KNN model
    distances, indices = knn.kneighbors(target_stats_df)

    valid_indices = []
    seen_names = set()

    for i in indices[0]:
        similar_name = df.iloc[i]['Full Name']
        if similar_name != name_input and similar_name not in seen_names:
            valid_indices.append(i)
            seen_names.add(similar_name)
        if len(valid_indices) == 5:
            break
    
    # Show stats of the selected player and most similar players
    st.write(f"Target Player's Stats for {name_input}:")
    target_player_info = non_transform_df.iloc[index_input]
    st.write(target_player_info[['Full Name', 'year', *data.columns]])  # Display relevant columns

    # Display 5 most similar players
    similar_player_info = non_transform_df.iloc[valid_indices]
    st.write(f"5 most similar players to {name_input}:")
    st.write(similar_player_info[['Full Name', 'year', *non_transform_df.columns]])

# Streamlit flow
name_input, year_input, index_input = user_input()

if name_input and year_input and index_input is not None:
    similarity(name_input, year_input, index_input)