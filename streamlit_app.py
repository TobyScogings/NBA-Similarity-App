### Changes to this document were made after the academy finished. This work did not fall within the scope of my final capstone project but was done as a personal passion project.

import streamlit as st
import pandas as pd, numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load data (make sure these paths are correct)
df = pd.read_csv("model_data.csv")
non_transform_df = pd.read_csv("model_data_pre-transform.csv")
data = df.drop(columns=['player_id', 'Full Name', 'team_name', 'year'])


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
    all_valid_indices = [] #List to store all valid indices
    for i in indices[0]:
        similar_name = df.iloc[i]['Full Name']
        if similar_name != name_input and similar_name not in seen_names:  # Removed year filter here
            all_valid_indices.append(i)
            seen_names.add(similar_name)

    valid_indices = all_valid_indices[:5] 

    st.write(f"Target Player's Stats for {name_input} in {year_input}:")
    target_player_info = non_transform_df.iloc[index_input].to_frame().T  # Convert to DataFrame
    st.dataframe(target_player_info[['Full Name', 'year', *data.columns]].style.format(precision=2), hide_index=True)  # Now it works

    # Percentiles Bar Chart
    
    df_percentiles = df.drop(columns=['player_id', 'Full Name', 'team_name', 'year']).rank(pct=True) * 100
    player_percentiles = df_percentiles.iloc[index_input]

    # Plot bar chart for each stat
    fig, ax = plt.subplots(figsize=(2, 3))  

    # Plot the player's percentile rankings
    player_percentiles.plot(kind='bar', ax=ax, colormap='viridis')
    
    # Graph Formatting
    ax.set_title(f"{name_input}'s Percentile Rankings ({year_input})")
    ax.set_xlabel('Stat')
    ax.set_ylabel('Percentile (%)')
    ax.set_xticklabels(player_percentiles.index, rotation=45)
    ax.set_ylim(0, 100)  # Ensure percentiles stay within the range
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    if valid_indices:
        similar_player_info = non_transform_df.iloc[valid_indices].drop(columns=['player_id'])
        st.write(f"5 most similar players to {name_input} in {year_input}:")
        st.dataframe(similar_player_info.style.format(precision=2), hide_index=True)  # Round for display
    else:
        st.write(f"No similar players found for {name_input} in {year_input}")

# Model Initialization
knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
knn.fit(data)

# --- Streamlit UI ---
st.title("Player Similarity App")  # Add a title

# Year selection
years = sorted(df['year'].unique().tolist(), reverse=True)
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