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

   
3.	Determined the number of unique values for each column.
   
  	<img src="https://github.com/Mago281/deep-learning-challenge/assets/131424690/77f605a2-8eed-476c-b18f-3bc1725c9890" width="200" height="175">


4.	For columns that had more than 10 unique values, determined the number of data points for each unique value.


5.	Used the number of data points for each unique value to pick a cutoff point to bin _"rare"_ categorical variables together in a new value, `Other`, and then checked if the binning was successful.

    <img src="https://github.com/Mago281/deep-learning-challenge/assets/131424690/2c834a4b-93d4-415d-97b4-ec544f102947" width="200" height="155">


6.	Used _pd.get_dummies()_ to encode categorical variables.


7.	Split the preprocessed data into a features array, `X`, and a target array, `y`.  Used these arrays and the _`train_test_split`_ function to split the data into training and testing datasets.


8.	Scaled the training and testing features datasets by creating a _StandardScaler_ instance, fitting it to the training data, then using the transform function.


________________________________________

### Step 2: *Compiled, Trained, and Evaluated the Model*

Using TensorFlow, I designed a deep learning model, to create a binary classification model that could predict if an Alphabet Soup-funded organisation would be successful based on the features in the dataset.  

To determine the number of neurons and layers in my model, I took into consideration how many inputs there were before determining the number of neurons and layers in my model.  Once I completed this step, I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

1.	Continued using the Jupyter Notebook in which i performed the preprocessing steps from Step.


2.	Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

  	 <img src="https://github.com/Mago281/deep-learning-challenge/assets/131424690/f97731f8-df33-43eb-8384-e696c162cc51" width="350" height="180">
    

3.	Created the first hidden layer and chose an appropriate activation function.


5.	Add a second hidden layer with an appropriate activation function.


6.	Created an output layer with an appropriate activation function.


7.	Checked the structure of the model.


8.	Compiled and trained the model, refer to the snapshot of the first 5 records appended below:

  	<img src="https://github.com/Mago281/deep-learning-challenge/assets/131424690/341bef03-f534-4069-aea4-9063efa646cb" width="550" height="170">


10.	#### Created a callback that saved the model's weights every five epochs.


11.	Evaluated the model using the test data to determine the loss and accuracy.


12.	Saved and exported the results to an HDF5 file and named the file **`AlphabetSoupCharity.h5`**.


________________________________________

### Step 3: *Optimised the Model*


Used any or all of the following methods to optimise the model:


**Note:  _I made three attempts at optimising my model to ensure that my model achieves target performance._**

1.	Created a new Jupyter Notebook file and name it _AlphabetSoupCharity_Optimisation.ipynb_.
    

2.	Imported dependencies and read in the _charity_data.csv_ to CoLab.
    

3.	Repeated the preprocessing steps above in a new notebook and used TensorFlow to optimise the model in order to achieve a target predictive accuracy higher than 75%.  


4.  Adjusted for any modifications that came out of optimising the model i.e. adjusted the input data to ensure that no variables or outliers would cause confusion in the model, e.g. dropping only 1 column (**`EIN`**) instead of 2 (**`EIN`** & **`NAME`**).
    

5.	Designed a neural network model, and adjusted it for modifications that would optimise the model to achieve higher than 75% accuracy.
    

6.	Saved and exported the results to an HDF5 file and named the file _AlphabetSoupCharity_Optimisation.h5_.
    

________________________________________

### Step 4: *Wrote a Report on the Neural Network Model*

Wrote a report on the performance of the deep learning model that I created for Alphabet Soup.  **Please refer to the following link below for the report that summarises the overall results of the deep learning model which includes a recommendation for how a different model could solve this classification problem, with an explanation for my recommendation:**


________________________________________


