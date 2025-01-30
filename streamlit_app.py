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
    name_input = st.text_input("Start typing a player name:")

    # Filter the list of players based on the text input
    filtered_players = df['Full Name'][df['Full Name'].str.contains(name_input, case=False, na=False)].unique().tolist() if name_input else []

    player_selection = None
    if filtered_players:
        # Create a selectbox with the filtered players
        player_selection = st.selectbox("Select a player", filtered_players)
        st.write(f"You selected: {player_selection}")
    else:
        st.write("No matching players found")
    
    # Add a unique key to the year_input element
    year_input = st.number_input("Select a year", min_value=2020, max_value=2024, step=1, value=2024, key="year_input")
    
    # Check if the input is valid (player exists and the year is within the allowed range)
    if st.button("Find Similar Players"):
        if player_selection not in df['Full Name'].values or year_input not in df['year'].values:
            st.error("This player or year is not valid, please try again.")
        else:
            # Get the index of the player
            index_input = df.index.get_loc(df[(df['Full Name'] == player_selection) & (df['year'] == year_input)].index[0])
            return player_selection, year_input, index_input

    st.write(f"Cleaned name input: '{name_input_clean}'")
    st.write(f"Cleaned year input: {year_input_clean}")
    
    # Return None if no valid input is provided
    return None, None, None

def similarity(name_input, year_input, index_input):
    if not name_input or not year_input or index_input is None:
        st.error("Invalid input. Please make sure to select a player and year.")
        return

    # Reshape the target's stats into the format needed for the KNN model
    target_stats = data.iloc[index_input].values.reshape(1, -1)
    target_stats_df = pd.DataFrame(target_stats, columns=data.columns)

    # Find the nearest datapoints and their respective distances to the user input
    distances, indices = knn.kneighbors(target_stats_df)

    # Create a list that holds the valid indices of similar players
    valid_indices = []

    # Create a set that stores players that we already have in our similarity output
    seen_names = set()

    # Iterate through every index, in order, that we found to be similar 
    for i in indices[0]:
        similar_name = df.iloc[i]['Full Name']
        if similar_name != name_input and similar_name not in seen_names:
            valid_indices.append(i)
            seen_names.add(similar_name)
        if len(valid_indices) == 5:
            break

    st.write(f"Target Player's Stats for {name_input}:")
    target_player_info = non_transform_df.iloc[index_input]
    st.write(target_player_info[['Full Name', 'year', *data.columns]])  # Display relevant columns

    # Display the 5 most similar players
    similar_player_info = non_transform_df.iloc[valid_indices]
    st.write(f"5 most similar players to {name_input}:")
    st.write(similar_player_info[['Full Name', 'year', *non_transform_df.columns]])

# Streamlit flow
name_input, year_input, index_input = user_input()

# If the inputs are valid, run the similarity function
if name_input and year_input and index_input is not None:
    similarity(name_input, year_input, index_input)