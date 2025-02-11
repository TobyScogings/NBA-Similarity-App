### Changes to this document were made after the academy finished. This work did not fall within the scope of my final capstone project but was done as a personal passion project.

### Created Jan/Feb 2025
### Data Sourced Jan. 28th 2025
### Developed by Toby Scogings

import streamlit as st
import pandas as pd, numpy as np
from sklearn.neighbors import NearestNeighbors
import altair as alt
from sklearn.preprocessing import StandardScaler

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

### Player Similarity Function
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
            percentile_df_for_chart['Actual Value'] = target_player_row[data.columns].values[0]

            percentile_df_for_chart['Percentile'] = percentile_df_for_chart['Percentile'].apply(lambda x: round(x))
            percentile_df_for_chart['Actual Value'] = percentile_df_for_chart['Actual Value'].apply(lambda x: round(x, 2))
                
            chart = alt.Chart(percentile_df_for_chart).mark_bar().encode(
            x=alt.X('Percentile:Q', title='Percentile'),
            y=alt.Y('Stat:N', sort=data.columns.tolist(), title='Stat'),
            color=alt.Color('Percentile:Q', scale=alt.Scale(domain=[0, 100], range=['red', 'green']), legend=None),
                        
            tooltip=[
            alt.Tooltip('Stat:N', title='Stat'),
            alt.Tooltip('Percentile:Q', title='Percentile'),
            alt.Tooltip('Actual Value:Q', title='Actual Stat Value')
            ]
            
            ).properties(
            title=f'Percentiles for {name_input} in {year_input}.',
            width=600
            )
            st.altair_chart(chart, use_container_width=True)
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
                percentile_df_for_chart['Actual Value'] = target_player_row[data.columns].values[0]
                
                percentile_df_for_chart['Percentile'] = percentile_df_for_chart['Percentile'].apply(lambda x: round(x))
                percentile_df_for_chart['Actual Value'] = percentile_df_for_chart['Actual Value'].apply(lambda x: round(x, 2))
                
                chart = alt.Chart(percentile_df_for_chart).mark_bar().encode(
                x=alt.X('Percentile:Q', title='Percentile'),
                y=alt.Y('Stat:N', sort=data.columns.tolist(), title='Stat'),
                color=alt.Color('Percentile:Q', scale=alt.Scale(domain=[0, 100], range=['red', 'green']), legend=None),
                tooltip=[
                alt.Tooltip('Stat:N', title='Stat'),
                alt.Tooltip('Percentile:Q', title='Percentile'),
                alt.Tooltip('Actual Value:Q', title='Actual Stat Value')
                ]
                ).properties(
                title=f'Percentiles for {name_input} this season.',
                width=600
                )
                st.altair_chart(chart, use_container_width=True)
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


### STAT COMPARISON


def stat_similarity(filled_columns, df, non_transform_df, user_input_df, comp_df):
    # Create a DataFrame using only the columns that were filled in by the user

    stat_data = data[filled_columns]
    
    # Initialize Nearest Neighbors model
    stat_knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
    
    # Fit the model with the comparison DataFrame
    stat_knn.fit(stat_data)
    
    # Now that the model is fitted, find the most similar players
    stat_similar_indices = find_similar_players(user_input_df, stat_knn, non_transform_df)

    if stat_similar_indices:
        stat_similar_player = non_transform_df.iloc[stat_similar_indices].drop(columns=['player_id'])
        st.write(f"5 most similar players to your custom statline (Note: Each player will only show once with their closest season!)")
        st.dataframe(stat_similar_player.style.format(precision=2), hide_index=True)
    else:
        st.write(f"No similar players to your custom statline found.")

def find_similar_players(user_input_df, stat_knn, non_transform_df):
    # Get distances and indices for the nearest neighbors
    distances, indices = stat_knn.kneighbors(user_input_df)

    valid_indices = []
    seen_players = set()  # Track player names

    # Loop through the indices of the nearest neighbors
    for i in indices[0]:
        player_name = non_transform_df.iloc[i]['Full Name']  # Get player's name

        if player_name not in seen_players:  # Ensure uniqueness
            seen_players.add(player_name)
            valid_indices.append(i)

        if len(valid_indices) == 5:  # Stop once we have 5 unique players
            break

    return valid_indices


### Player Comparison Option


def player_comp(df):
    # Year selection
    years = sorted(df['year'].unique().tolist(), reverse=True)
    default_year = max(years) if years else None  # Default year handling if the list is empty
    selected_year = st.selectbox("Select Year", years, index=years.index(default_year) if default_year in years else 0) if years else None
    
    # Player selection (dynamically populated based on selected year)
    if selected_year is not None:  # Checks if a year has been selected
        players_in_year = df[df['year'] == selected_year].sort_values(by='Full Name')['Full Name'].tolist()
    
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

def stat_comp(non_transform_df):
    
    st.subheader("Enter Your Custom Statline")

    # Mandatory stat entries
    points = st.slider("Points Per Game", min_value=0.01, max_value = max(non_transform_df['Points']), step=0.01)
    assists = st.slider("Assists Per Game", min_value=0.01, max_value = max(non_transform_df['Assists']), step=0.01)
    rebounds = st.slider("Rebounds Per Game", min_value=0.01, max_value = max(non_transform_df['Rebounds']), step=0.01)
    steals = st.slider("Steals Per Game", min_value=0.01, max_value = max(non_transform_df['Steals']), step=0.01)
    blocks = st.slider("Blocks Per Game", min_value=0.01, max_value = max(non_transform_df['Blocks']), step=0.01)

    # Optional stat entries
    if min(points, assists, rebounds, steals, blocks) == 0:
        st.write("You must first set your values for these 5 key stats")
    else:
        st.subheader(f"\nPlease now choose any other stats you would like to add in:\n")
        
        optional_stats = {'min': 'Minutes',
        'fga': 'FGA',
        'fg%': 'FG%',
        'tpa': '3PA',
        'tp%': '3P%',
        'fta': 'FTA',
        'ft%': 'FT%',
        'defReb': 'Def. Rebounds',
        'offReb': 'Off. Rebounds',
        'pFouls': 'Fouls',
        'turnovers': 'Turnovers'}
    
        selected_stats = {}
        
        # Loop to create checkboxes dynamically
        for key, label in optional_stats.items():
            if st.checkbox(label):  # Checkbox with stat name
                selected_stats[key] = st.slider(f"Enter {label}", min_value=0.01, max_value=max(non_transform_df[label]), step=0.1, value=0.0)
    
        # if selected_stats:
        #     for label, value in selected_stats.items():
        #         st.write(f"- **{label}**: {round(value,2)}")
        # else:
        #     st.write("No additional stats selected.")
    
        
        
        if st.button("Find my comparisons!"):
            # Collect all input data into a dictionary
            input_data = {
                'Points': points,
                'Assists': assists,
                'Rebounds': rebounds,
                'Steals': steals,
                'Blocks': blocks
            }
    
            # Add selected optional stats to the dictionary
            for key, value in selected_stats.items():
                input_data[optional_stats[key]] = value

            input_df = pd.DataFrame([input_data])

            st.subheader("Your stat inputs are:")
            st.write(input_df)

###############################################################################################################  --- Input Log Transformation ---

            # Define the columns the user has selected (all non-zero columns)
            # Remove percentage columns
            # Apply log transformation
            # This all follows the method used in data preparation
            
            filled_columns = [col for col, val in input_data.items() if val != 0.0]
            
            transform_input = filled_columns.copy()
            if 'FG%' in filled_columns:
                transform_input.remove('FG%')
            if '3P%' in filled_columns:
                transform_input.remove('3P%')
            if 'FT%' in filled_columns:
                transform_input.remove('FT%')



            input_df[transform_input] = input_df[transform_input].apply(lambda x: np.log(x + 0.0001))


###############################################################################################################  --- Input Scaling ---
            
            # Columns to scale: filled_columns
            # Create extract of non_transform_df with cols from filled_columns
            # Log our original data in order to define the scaler for our singular row
            # Initialise and fit the scaler on this full dataset
            # Apply this scaler to transform our singular input
            
            comp_df = non_transform_df[filled_columns]

            df_logged = non_transform_df[transform_input].apply(lambda x: np.log(x + 0.0001))
            other_cols = non_transform_df.drop(columns=transform_input)
            df_combined = pd.concat([df_logged, other_cols], axis=1)
            df_combined = df_combined[filled_columns]

            df_scaled = df_combined.copy()

            scaler = StandardScaler()
            df_scaled[filled_columns] = scaler.fit_transform(df_combined[filled_columns])

            input_df_scaled = input_df.copy()
            input_df_scaled[filled_columns] = scaler.transform(input_df[filled_columns])

            user_input_df = input_df_scaled


################################################################################################################
            

            stat_similarity(filled_columns, df, non_transform_df, user_input_df, comp_df)




################################################################################################################ --- Streamlit UI ---


st.title("NBA Player Similarity App")  

st.subheader("What does this app do?")

st.write("""This app allows users to compare player stats and find similar players. Current functionality allows users to select any player that has min. 1 second of playing time since 2020/21 and find their 5 most similar players. First, select the year to compare and then the desired player.

Future updates include the ability to compare custom statlines, standardise stats per 36 minutes and more!""")

with st.sidebar:
    st.subheader("NBA Comparison Tool")

if "active_feature" not in st.session_state:
    st.session_state.active_feature = "player_comp"

# Create system choice buttons
choice_col1, choice_col2 = st.columns(2)

with choice_col1:
    if st.button("🏀 Player Comparison", key="player_button"):
        st.session_state.active_feature = "player_comp"

with choice_col2:
    if st.button("📊 Statline Comparison", key="stat_button"):
        st.session_state.active_feature = "stat_comp"

# Check session state and run the selected function
if st.session_state.active_feature == "player_comp":
    player_comp(df)
elif st.session_state.active_feature == "stat_comp":
    stat_comp(non_transform_df)