###############################################################################################################  --- Per 36 Function ---    


# def p36_similarity(name_input, year_input, index_input):
#     if not name_input or not year_input or index_input is None:
#         st.error("Invalid input. Please make sure to select a player and year.")
#         return

#     target_stats = data.iloc[index_input].values.reshape(1, -1)
#     target_stats_df = pd.DataFrame(target_stats, columns=data.columns)


# ###############################################################################################################  --- Modelling Output Definitions ---
    
#     distances, indices = knn.kneighbors(target_stats_df)

#     valid_indices = []
#     seen_names = set()
#     all_valid_indices = []
#     for i in indices[0]:
#         similar_name = df.iloc[i]['Full Name']
#         if similar_name != name_input and similar_name not in seen_names:
#             all_valid_indices.append(i)
#             seen_names.add(similar_name)

#     valid_indices = all_valid_indices[:5]


# ###############################################################################################################  --- Chosen Player Stats ---    

#     st.write(f"Target Player's Stats for {name_input} in {year_input}:")
#     target_player_info = non_transform_df.iloc[index_input].to_frame().T
#     st.dataframe(target_player_info[['Full Name', 'year', *data.columns]].style.format(precision=2), hide_index=True)

#     col1, col2 = st.columns(2)
    
# ###############################################################################################################  --- Chosen Season Percentile Graph ---    
    
#     with col1:
#         year_data = non_transform_df[non_transform_df['year'] == year_input]
#         percentile_df_year = year_data[data.columns].apply(lambda x: x.rank(pct=True) * 100)
    
#         # ***Correct way to get the percentile***
#         target_player_row = non_transform_df[(non_transform_df['Full Name'] == name_input) & (non_transform_df['year'] == year_input)]
#         if not target_player_row.empty: #Check to make sure the row exists
#             target_player_index_year = target_player_row.index[0]
#             target_player_percentile = percentile_df_year.loc[target_player_index_year] #Use .loc to access the percentile by index
                
#             percentile_df_for_chart = target_player_percentile.to_frame(name="Percentile")
#             percentile_df_for_chart['Stat'] = percentile_df_for_chart.index
#             percentile_df_for_chart['Actual Value'] = target_player_row[data.columns].values[0]

#             percentile_df_for_chart['Percentile'] = percentile_df_for_chart['Percentile'].apply(lambda x: round(x))
#             percentile_df_for_chart['Actual Value'] = percentile_df_for_chart['Actual Value'].apply(lambda x: round(x, 2))
                
#             chart = alt.Chart(percentile_df_for_chart).mark_bar().encode(
#             x=alt.X('Percentile:Q', title='Percentile'),
#             y=alt.Y('Stat:N', sort=data.columns.tolist(), title='Stat'),
#             color=alt.Color('Percentile:Q', scale=alt.Scale(domain=[0, 100], range=['red', 'green']), legend=None),
                        
#             tooltip=[
#             alt.Tooltip('Stat:N', title='Stat'),
#             alt.Tooltip('Percentile:Q', title='Percentile'),
#             alt.Tooltip('Actual Value:Q', title='Actual Stat Value')
#             ]
            
#             ).properties(
#             title=f'Percentiles for {name_input} in {year_input}.',
#             width=600
#             )
#             st.altair_chart(chart, use_container_width=True)
#         else:
#             st.write(f"No data found for {name_input} in {year_input} to calculate percentiles.")

    
# ###############################################################################################################  --- 2024 Percentile Graph ---
    
#     if year_input != 2024 and df[(df['Full Name'] == name_input) & (df['year'] == 2024)].empty == False:
#         with col2:
#             year_data = non_transform_df[non_transform_df['year'] == 2024]
#             percentile_df_year = year_data[data.columns].apply(lambda x: x.rank(pct=True) * 100)
        
#             # ***Correct way to get the percentile***
#             target_player_row = non_transform_df[(non_transform_df['Full Name'] == name_input) & (non_transform_df['year'] == 2024)]
#             if not target_player_row.empty: #Check to make sure the row exists
#                 target_player_index_year = target_player_row.index[0]
#                 target_player_percentile = percentile_df_year.loc[target_player_index_year] #Use .loc to access the percentile by index
        
#                 percentile_df_for_chart = target_player_percentile.to_frame(name="Percentile")
#                 percentile_df_for_chart['Stat'] = percentile_df_for_chart.index
#                 percentile_df_for_chart['Actual Value'] = target_player_row[data.columns].values[0]
                
#                 percentile_df_for_chart['Percentile'] = percentile_df_for_chart['Percentile'].apply(lambda x: round(x))
#                 percentile_df_for_chart['Actual Value'] = percentile_df_for_chart['Actual Value'].apply(lambda x: round(x, 2))
                
#                 chart = alt.Chart(percentile_df_for_chart).mark_bar().encode(
#                 x=alt.X('Percentile:Q', title='Percentile'),
#                 y=alt.Y('Stat:N', sort=data.columns.tolist(), title='Stat'),
#                 color=alt.Color('Percentile:Q', scale=alt.Scale(domain=[0, 100], range=['red', 'green']), legend=None),
#                 tooltip=[
#                 alt.Tooltip('Stat:N', title='Stat'),
#                 alt.Tooltip('Percentile:Q', title='Percentile'),
#                 alt.Tooltip('Actual Value:Q', title='Actual Stat Value')
#                 ]
#                 ).properties(
#                 title=f'Percentiles for {name_input} this season.',
#                 width=600
#                 )
#                 st.altair_chart(chart, use_container_width=True)
#             else:
#                 st.write(f"No data found for {name_input} in {year_input} to calculate percentiles.")


# ###############################################################################################################  --- Similar Players Output ---    
    
#     if valid_indices:
#         similar_player_info = non_transform_df.iloc[valid_indices].drop(columns=['player_id'])
#         st.write(f"5 most similar players to {name_input} in {year_input}:")
#         st.dataframe(similar_player_info.style.format(precision=2), hide_index=True)
#     else:
#         st.write(f"No similar players found for {name_input} in {year_input}")

# # Model Initialization
# knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
# knn.fit(data)

###############################################################################################################  --- Player Per 36 Comparison ---


# def player_comp_p36(df):

#     st.write("This functionality is a work in progress. Please check back later. For now, feel free to use the other functions available")
    
#     # Year selection
#     years = sorted(p36['year'].unique().tolist(), reverse=True)
#     default_year = max(years) if years else None  # Default year handling if the list is empty
#     selected_year = st.selectbox("Select Year", years, index=years.index(default_year) if default_year in years else 0) if years else None
    
#     # Player selection (dynamically populated based on selected year)
#     if selected_year is not None:  # Checks if a year has been selected
#         players_in_year = p36[p36['year'] == selected_year].sort_values(by='Full Name')['Full Name'].tolist()
    
#         if players_in_year:
#             selected_player = st.selectbox("Select Player", players_in_year)
    
#             try:
#                 index_input = p36[(p36['Full Name'] == selected_player) & (p36['year'] == selected_year)].index[0]
#                 #similarity(selected_player, selected_year, index_input)
    
#             except IndexError:
#                 st.error(f"No data found for {selected_player} in {selected_year}. Please select a different player or year.")
    
#         else:
#             st.write(f"No players found for the year {selected_year}")
#     else:
#         st.write("No years available in the data.")

#     st.write(f"Year: {selected_year}, Player: {selected_player}, index: {index_input}")
