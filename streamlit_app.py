### Changes to this document were made after the academy finished. This work did not fall within the scope of my final capstone project but was done as a personal passion project.

import streamlit as st
import pandas as pd, numpy as np
from sklearn.neighbors import NearestNeighbors
import altair as alt

st.set_page_config(layout="wide")

# Load data (make sure these paths are correct)
df = pd.read_csv("model_data.csv")
non_transform_df = pd.read_csv("model_data_pre-transform.csv")

# Column renaming

df = df.rename(columns={
    'points': 'Points',
    'totReb': 'Rebounds',
    'assists': 'Assists',
    'steals': 'Steals',
    'blocks': 'Blocks',
    'min': 'Minutes',
    'fga': 'FGA',
    'fg%': 'FG%',
    'tpa': '3PA',
    'tp%': '3P%',
    'fta': 'FTA',
    'ft%': 'FT%',
    'defReb': 'Def. Rebounds',
    'offReb': 'Off. Rebounds',
    'pFouls': 'Fouls',
    'turnovers': 'Turnovers'
})

non_transform_df = non_transform_df.rename(columns={
    'points': 'Points',
    'totReb': 'Rebounds',
    'assists': 'Assists',
    'steals': 'Steals',
    'blocks': 'Blocks',
    'min': 'Minutes',
    'fga': 'FGA',
    'fg%': 'FG%',
    'tpa': '3PA',
    'tp%': '3P%',
    'fta': 'FTA',
    'ft%': 'FT%',
    'defReb': 'Def. Rebounds',
    'offReb': 'Off. Rebounds',
    'pFouls': 'Fouls',
    'turnovers': 'Turnovers'
})

data = df.drop(columns=['player_id', 'Full Name', 'team_name', 'year'])
# Percentiles Bar Chart
custom_order = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Minutes', 'FGA', 'FG%', '3PA', '3P%', 'FTA', 'FT%', 'Def. Rebounds', 'Off. Rebounds', 'Fouls', 'Turnovers']

# Calculate the percentiles
percentile_df = non_transform_df[custom_order].apply(lambda x: x.rank(pct=True) * 100)

# --- Similarity Function ---
def similarity(name_input, year_input, index_input):
    if not name_input or not year_input or index_input is None:
        st.error("Invalid input. Please make sure to select a player and year.")
        return

    target_stats = data.iloc[index_input].values.reshape(1, -1)
    target_stats_df = pd.DataFrame(target_stats, columns=data.columns)


    ### MODELLING STAGE AND OUTPUT DEFINITIONS

    
    distances, indices = knn.kneighbors(target_stats_df)

    valid_indices = []
    seen_names = set()
    all_valid_indices = []
    for i in indices[0]:
        similar_name = df.iloc[i]['Full Name']
        if similar_name != name_input and similar_name not in seen_names:
            all_valid_indices.append(i)
            seen_names.add(similar_name)

    valid_indices = all_valid_indices[:5]


    ### SELECTED PLAYER CURRENT STATS
    

    st.write(f"Target Player's Stats for {name_input} in {year_input}:")
    target_player_info = non_transform_df.iloc[index_input].to_frame().T
    st.dataframe(target_player_info[['Full Name', 'year', *data.columns]].style.format(precision=2), hide_index=True)

    col1, col2 = st.columns(2)
    
    ### SELECTED SEASON PERCENTILES
    
    with col1:
        year_data = non_transform_df[non_transform_df['year'] == year_input]
        percentile_df_year = year_data[data.columns].apply(lambda x: x.rank(pct=True) * 100)
    
        # ***Correct way to get the percentile***
        target_player_row = non_transform_df[(non_transform_df['Full Name'] == name_input) & (non_transform_df['year'] == year_input)]
        if not target_player_row.empty: #Check to make sure the row exists
          target_player_index_year = target_player_row.index[0]
          target_player_percentile = percentile_df_year.loc[target_player_index_year] #Use .loc to access the percentile by index
    
          percentile_df_for_chart = target_player_percentile.to_frame(name="Percentile")
          percentile_df_for_chart['Stat'] = percentile_df_for_chart.index
    
          chart = alt.Chart(percentile_df_for_chart).mark_bar().encode(
            x=alt.X('Percentile:Q', title='Percentile'),
            y=alt.Y('Stat:N', sort=data.columns.tolist(), title='Stat'),
            color=alt.Color('Percentile:Q', scale=alt.Scale(domain=[0, 100], range=['red', 'green']), legend=None),
          ).properties(
            title=f'Percentiles for {name_input} in {year_input}.',
            width=600
          )
          st.altair_chart(chart, use_container_width=False)
        else:
          st.write(f"No data found for {name_input} in {year_input} to calculate percentiles.")

    
    ### ALL TIME PERCENTILES
    if year_input != 2024 and df[(df['Full Name'] == name_input) & (df['year'] == 2024)].empty == False:
        with col2:
            year_data = non_transform_df[non_transform_df['year'] == 2024]
            percentile_df_year = year_data[data.columns].apply(lambda x: x.rank(pct=True) * 100)
        
            # ***Correct way to get the percentile***
            target_player_row = non_transform_df[(non_transform_df['Full Name'] == name_input) & (non_transform_df['year'] == 2024)]
            if not target_player_row.empty: #Check to make sure the row exists
              target_player_index_year = target_player_row.index[0]
              target_player_percentile = percentile_df_year.loc[target_player_index_year] #Use .loc to access the percentile by index
        
              percentile_df_for_chart = target_player_percentile.to_frame(name="Percentile")
              percentile_df_for_chart['Stat'] = percentile_df_for_chart.index
        
              chart = alt.Chart(percentile_df_for_chart).mark_bar().encode(
                x=alt.X('Percentile:Q', title='Percentile'),
                y=alt.Y('Stat:N', sort=data.columns.tolist(), title='Stat'),
                color=alt.Color('Percentile:Q', scale=alt.Scale(domain=[0, 100], range=['red', 'green']), legend=None),
              ).properties(
                title=f'Percentiles for {name_input} this season.',
                width=600
              )
              st.altair_chart(chart, use_container_width=False)
            else:
              st.write(f"No data found for {name_input} in {year_input} to calculate percentiles.")


    ### SIMILAR PLAYERS OUTPUTS
    
    
    if valid_indices:
        similar_player_info = non_transform_df.iloc[valid_indices].drop(columns=['player_id'])
        st.write(f"5 most similar players to {name_input} in {year_input}:")
        st.dataframe(similar_player_info.style.format(precision=2), hide_index=True)
    else:
        st.write(f"No similar players found for {name_input} in {year_input}")

# Model Initialization
knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
knn.fit(data)


### Player Comparison Option

def player_comp(df):
    # Year selection
    years = sorted(df['year'].unique().tolist(), reverse=True)
    default_year = max(years) if years else None  # Default year handling if the list is empty
    selected_year = st.selectbox("Select Year", years, index=years.index(default_year) if default_year in years else 0) if years else None
    
    # Player selection (dynamically populated based on selected year)
    if selected_year is not None:  # Checks if a year has been selected
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


### --- Streamlit UI ---


st.title("Player Similarity App")  # Title

st.subheader("What does this app do?")

choice_col1, choice_col2 = st.columns(2) # Create system choice buttons

with choice_col1:
    player_selected = st.button("🏀 Player Comparison", key="player_button")

with choice_col2:
    stat_selected = st.button("📊 Statline Comparison", key="stat_button")

if player_selected:
    player_comp(df)
if stat_selected:
    st.write("Statline Comparison Selected! This feature is a work in progress. For now, please return to player comparison!")