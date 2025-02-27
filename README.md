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

- **HTML Exploration**: Began with a look into HTML tags, webpage structure and key information to gather
- **Homepage Scraping**: First step in scraping was to work on the homepage. Learnt how to extract text from tags and used this to gather link suffixes which were reverse engineered into workable links.
- **Python Understanding**: Looked into manipulating what is returned by BeautifulSoup web scraping and used this to create a preliminary dataframe holding all books on the site
- **Output Generation**: Scraped each book listing page individually to get an extensive dataframe holding all information related to each book. Exported this dataframe to a csv for further use.
- **Project Explanation**: Went over the code, explaining techniques through commetns and cleaning up code via function division.

## Outcomes 🎉

*Technical*:

- `Scraped Books.csv` is the final output from my code. It gives each books title, hyperlink and a range of information relating to it in a csv format
- Develop the basis of a web scraping tool. It fulfills the task as intended however has room for improvement should I find the time. These improvements are described at the beginning of `BooksToScrape_Book_Scraper`.

*Personal*:

- Learn more about webpages and how they are coded and structured
- Practice web scraping through python
- Strengthen basic python techniques including list/dictionary manipulation, looping and function relationships

# Future Changes

- Improved output comparisons
- Improved input support - allow filtering of player options.
- Per minute or per 36 minutes stat option
- Improved system explanation on site
- Output filtering?
