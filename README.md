# 🏀 NBA Player Comparison Tool 🏀 

Data Analytics training project converted into personal development passion project. An app built to allow for comparison of NBA players' seasons since 2020

### Access my application [here](https://nbasimilarity.streamlit.app/)

## About the Project 🌟

This project began as part of my data analyst training and developed into a personal project developed not only to enhance my skills within data analytics, but also to create a platform that allows me and others to elevate their basketball knowledge. This was developed to create the possibility of cross-comparison of statlines and to allow viewers of the sport to contextualise the bombardment of stats that often occurs on basketball broadcasts.

Ultimately, the outcome so far has been an application that takes any player to have played NBA minutes since 2020/21 and returns the 5 most statistically similar seasons using a K-nearest neighbour model. Additionally, custom statline functionality has been implemented, finding the 5 most similar seasons to custom inputs using the same method.

## Goals 🎯

- Understand Project Management:
  - Experience every stage of a project, from creation to implementation and documentation
  - Understand the importance of agile workflows, creating fast and frequent additions
- Learn about API Scraping and the power it holds in data acquisition
- Reinforce data preparation skills including, but not limited to:
  - Null Handling
  - Data Transformations
  - Type Standardisation
- Implement different modelling methods to evaluate pros and cons before deciding on an optimal method
- Explore application hosting through Streamlit.
- Iterate upon success to continue to build a platform to elevate the statistical understanding of basketball fans

## Technologies Used 🖥️

- API - [RapidAPI's 'API-NBA'](https://rapidapi.com/api-sports/api/api-nba)
- `python`
  - `pandas`
  - `numpy`
  - `altair`
  - `sklearn`
- Streamlit
- Microsoft Excel CSV Files (for data exportation)
  - `model_data.csv`
  - `model_data_pre-transform.csv`
  - `per_36.csv`
  - `reg_per_36.csv`
 




## Project Structure 🔀

- **Problem Framing**: Began with devising a problem. Inspired when watching basketball and noticing low contextualisation of stats I was reading.
- **Data Acquisition**: Investigate different methods of data collection. Looked into web scraping and subscription to databases but these had their limitations (ethical consideration, webpage slowing, T&Cs etc). Found an API to use which allowed me to extarct the necessary data (see above).
- **Data Exploration**: Understand the data that I am working with. What format is time data in, what desciptive data have I pulled and what scale is my data on. Includes data visulation of distributions, correlations etc.
- **Data Preparation**: Fix different formats of time data to standardise the column. Apply a log transformation to fix the right skew due to 0 starting values at the begining of games. Create aggregate featuers such as shot percentages.
- **Model Practicing**: Attempt modelling methods to find the optimal method. Began using GMM to create clusters of similar players but class sizes were wildly imbalanced - not optimal. Settle on K-Nearest Neighbour modelling to get distances between statlines to find the closest data points to each player.
- **Model Tuning**: Fix some issues with the initial model, including returning to the dataset to complete more feature engineering (e.g. applying standard scaling to standardise numerical columns). Create the optimal model to use to compare most similar statlines.
- **Application Hosting**: Use streamlit to validate user inputs and create interactivity with the system. Allow for dynamic outputs relevant to user inputs.
- **Project Presentation**: Demonstrate and explain this process to a live audience, showing the process from start to finish with a demonstration of how to system works and how I have used it to enhance my experience.
- **Application Improvement**: Post-completion of the project, develop new features and improve functionality. Some of these improvements include statline normalisation, custom statline comparison and various QoL and UI improvements.

