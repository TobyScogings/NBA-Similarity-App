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
    while True:
        name_input = st.text_input("Start typing a player name:")
        
        # Filter the list of cities based on user input
        filtered_players = df['Full Name'][df['Full Name'].str.contains(name_input, case=False, na=False)].tolist() if name_input else []
        
        # Show the filtered suggestions
        st.write("Suggestions:")
        for player in filtered_players:
            st.write(player)
        
        year_input = st.number_input("Select a year", min_value=2020, max_value=2024, step=1, value=2024)
        
        # Check if the input is valid (player exists and the year is within the allowed range)
        if st.button("Find Similar Players"):
        if name_input not in df['Full Name'].values or year_input not in df['year'].values:
            st.error("This player or year is not valid, please try again.")
        else:
            # Get the index of the player
            index_input = df.index.get_loc(df[(df['Full Name'] == name_input) & (df['year'] == year_input)].index[0])
            st.write(f"Name: {name_input}")
            st.write(f"Year: {year_input}")
            st.write(f"Row Index: {index_input}")

def similarity(name_input, year_input, index_input):

    # Reshape the targets stats into the format needed for the KNN model
    target_stats = data.iloc[index_input].values.reshape(1, -1)
    target_stats_df = pd.DataFrame(target_stats, columns = data.columns)

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
    target_player_info = df.iloc[index_input]
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
