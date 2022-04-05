## This repository contains the code for the clustering project completed as part of the Codeup Data Science curriculum. 

## Repo contents:
### 1. This Readme:
- Project description with goals
- Inital hypothesis/questions on data, ideas
- Data dictionary
- Project planning
- Instructions for reproducing this project and findings
- Key findings and recommendations for this project
### 2. Final report (final_report.ipynb)
### 3. Wrangle module (wrangle.py)
### 4. Exploration & modeling notebooks (prepare_explore_model.ipynb)
### 5. Functions to support exploration and modeling work (explore.py, model.py)

### Project Goals

The goal of this project was to identify key drivers of property value for single family properties. I used statistical testing and machine learning models to provide insight into what affects home prices.

### Initial Questions and Hypotheses

1. Does the number of bathrooms affect the home's tax value?
    - I suspect a greater number would increase the tax value
2. Does the number of bedrooms affect the home's tax value?
    - I suspect a greater number would increase the tax value
3. Does the square footage of the home affect tax value?
    - Larger homes are probably more valuable
4. Does the home's age affect the home's tax value?
    - I suspect older homes are less valuable but then again there are some historic homes that could be more valuable. 
5. How much does the location of the home affect home value?
    - I suspect this is a key driver. With this dataset I only have county data but believe more localized neighborhood data is important.
6. Does the ratio of bedrooms to bathroooms have any effect on tax value?
    - I'm not sure if this would have a very strong effect, but worth checking
7. Does whether a home includes a pool or garage have a significant effect on the home's tax value?
    - Having these additional features should increase the home's value
8. Does the home's living space, defined as the square footage not including bathrooms and bedrooms, affect the home's tax value?
    - More living space is likely more valuable


### Data Dictionary

| Variable    | Meaning     |
| ----------- | ----------- |
| bedroom    |  number of bedrooms in home         |
| bathroom           |  number of bathrooms in home          |
| bathroom_cat    |  number of bathrooms split into 3 categories     |
| bedroom_cat   |  number of bedrooms split into 3 categories     |
| age    |  age of the home   |
| square_feet    |  total living area of home    |
| tax_value           | total tax assessed value of the parcel (target) |
| fips    |  Federal Information Processing Standard code (location)       |
| bed_to_bath    |  ratio of bedrooms to bathrooms      |
| living_space   |  square footage - (bathrooms*40 + bedrooms*200)       |
| room_count    |  sum of bedrooms and bathrooms       |
| pool    |  whether the home has a pool      |
| has_garage   |  whether the home has a garage      |
| condition   |  assessment of the condition of the home, low values are better       |


### Project Plan

For this project I followed the data science pipeline:

Planning: I established the goals for this project and the relevant questions I wanted to answer. I developed a Trello board to help keep track of open and completed work (https://trello.com/b/mT9nMlHN/zillow-regression-project).

Acquire: The data for this project is from a SQL Database called 'zillow'. The wrangle.py script is used to query the database for the required data tables and returns the data in a Pandas DataFrame. This script also saves the DataFrame to a .csv file for faster subsequent loads. The script will check if the zillow_2017.csv file exists in the current directory and if so will load it into memory, skipping the SQL query.

Prepare: The wrangle.py script uses the same wrangle_zillow function from the acquire step to prepare the data for exploration and modeling. Steps here include removing null values (NaN), generating additional features, and encoding categorical variables. This script also contains a split_data function to split the dataset into train, validate, and test sets cleanly. Additional functions to remove outliers and scale the data are included in this file.

Explore: The questions established in planning were analyzed using statistical tests including correlation and t-tests to test hypotheses about the data. This work was completed in the prepare_explore_model.ipynb file and relevant portions were moved to the final_report.ipynb final deliverable. A visualization illustrating the results of the tests and answering each question is included. 

Model: Four different regression algorithms were investigated to determine if home tax values could be predicted using features identified during exploration. A select set of hyperparameters were tested against train and validate data to determine which demonstrated the best performance. The final model was selected and used to make predictions on the withheld test data.

Delivery: This is in the form of this github repository as well as a presentation of my final notebook to the stakeholders.

### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow database. Store that env file locally in the repository. 
2. Clone my repository (including the wrangle.py,explore.py, and model.py modules). Confirm .gitignore is hiding your env.py file.
3. Libraries used are pandas, matplotlib, scipy, sklearn, seaborn, and numpy.
4. You should be able to run final_report.ipynb.

### Key Findings and Recommendations

- Location, living space, and age were shown to be moderately correlated with tax value.
- Average tax value looks to be higher than the overall average for more than 3 bedrooms and lower for 3 or less:
- Average tax value looks to be higher than the overall average for more than 2 bathrooms and lower for 2 or less
- Seems to be a linear relationship between square feet and tax value
- Having a garage adds value to a home
- The greater the age the lower the value in general
- The county the home is in affects the tax value: LA looks to be less than other two
- Having a pool seems more valuable than not
- Several regression models were trained and evaluated to attempt to predict tax assessed values of single family properties. Improvements of over 20% over baseline were achieved although r2 and RMSE values were poor enough that I cannot recommend these models for production.
- Additional feature engineering work may help improve performance.
- Ultimately the dataset lacks key information that would likely be more predictive of tax assessed value, such as finer neighborhood information and home condition (although this is available for one county it was not shown to be particularly predictive). 

### Future work

- Explore other factors that may affect home values
- Create additional features for modeling
- Perform additional feature engineering work 
- Test additional hyperparameters for the models
