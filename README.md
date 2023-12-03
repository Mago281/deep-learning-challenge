# deep-learning-challenge

## BACKGROUND

The nonprofit foundation Alphabet Soup is in need of a tool that can help it select the applicants for funding with the best chance of success in their ventures.  With my knowledge of machine learning and neural networks, I will use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

Alphabet Soup’s business team provided me with a CSV containing more than 34,000 organisations that have received funding from Alphabet Soup over the years.  Within this dataset are a number of columns that capture metadata about each organisation, such as:

  - **EIN** and **NAME** — Identification columns

  - **APPLICATION_TYPE** — Alphabet Soup application type

  - **AFFILIATION** — Affiliated sector of industry

  - **CLASSIFICATION** — Government organisation classification

  - **USE_CASE** — Use case for funding

  - **ORGANIZATION** — Organisation type

  - **STATUS** — Active status

  - **INCOME_AMT** — Income classification

  - **SPECIAL_CONSIDERATIONS** — Special considerations for application

  - **ASK_AMT** — Funding amount requested

  - **IS_SUCCESSFUL** — Was the money used effectively

________________________________________

## STEPS TAKEN
________________________________________

### Step 1: *Preprocessing of the Data*

I used Pandas and scikit-learn’s StandardScaler() and performed the following steps to complete the preprocessing steps:

1.	Read in the charity_data.csv to a Pandas DataFrame, and identified the following in the dataset:

     ![image](https://github.com/Mago281/deep-learning-challenge/assets/131424690/7e09bf14-d865-462c-815b-26c08b7d5070)

    -  _What variable(s) are the target(s) for the model ?_
        #### 
    -  _What variable(s) are the feature(s) for the model ?_
        #### 

2.	Dropped the non-beneficial ID columns **EIN** and **NAME**.
    ![image](https://github.com/Mago281/deep-learning-challenge/assets/131424690/77f605a2-8eed-476c-b18f-3bc1725c9890)

      <img src="https://github.com/Mago281/CryptoClustering/assets/131424690/bd3f4052-6d7f-4594-b4dd-a425ff556df0" width="700" height="300">

4.	Determined the number of unique values for each column.


5.	For columns that have more than 10 unique values, determine the number of data points for each unique value.


6.	Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.


7.	Use _pd.get_dummies()_ to encode categorical variables.


8.	Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the _train_test_split_ function to split the data into training and testing datasets.


9.	Scale the training and testing features datasets by creating a _StandardScaler_ instance, fitting it to the training data, then using the transform function.


________________________________________

### Step 2: *Compile, Train, and Evaluate the Model*

Using TensorFlow, I designed a neural network, or deep learning model, to create a binary classification model that could predict if an Alphabet Soup-funded organisation would be successful based on the features in the dataset.  

To determine the number of neurons and layers in my model, I took into consideration how many inputs there were before determining the number of neurons and layers in my model.  Once I completed this step, I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.


1.	Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.


2.	Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.


3.	Create the first hidden layer and choose an appropriate activation function.


4.	If necessary, add a second hidden layer with an appropriate activation function.


5.	Create an output layer with an appropriate activation function.


6.	Check the structure of the model.


7.	Compile and train the model.


8.	Create a callback that saves the model's weights every five epochs.


9.	Evaluate the model using the test data to determine the loss and accuracy.


10.	Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.


________________________________________

### Step 3: *Optimise the Model*

Used TensorFlow to optimise the model to achieve a target predictive accuracy higher than 75%.

Used any or all of the following methods to optimise the model:

- Adjusted the input data to ensure that no variables or outliers would cause confusion in the model, such as:

      o	Dropping more or fewer columns.
  
      o	Creating more bins for rare occurrences in columns.
  
      o	Increasing/decreasing the number of values for each bin.
  
  
- Added more neurons to a hidden layer.

- Added more hidden layers.

- Used different activation functions for the hidden layers.

- Added/reduced the number of epochs to the training regimen.


***Note:  I made three attempts at optimising my model to ensure that my model achieves target performance.***

1.	Created a new Jupyter Notebook file and name it _AlphabetSoupCharity_Optimisation.ipynb_.
    

2.	Imported dependencies and read in the _charity_data.csv_ to CoLab.
    

3.	Pre-processed the dataset as in Step 1 and adjusted for any modifications that came out of optimising the model.
    

4.	Designed a neural network model, and adjustrd it for modifications that would optimise the model to achieve higher than 75% accuracy.
    

5.	Saved and exported the results to an HDF5 file and named the file _AlphabetSoupCharity_Optimisation.h5_.
    

________________________________________

### Step 4: *Wrote a Report on the Neural Network Model*

Wrote a report on the performance of the deep learning model that I created for Alphabet Soup.  Please refer to the following link below for the report that summarises the overall results of the deep learning model which includes a recommendation for how a different model could solve this classification problem, with an explanation for my recommendation:


________________________________________


