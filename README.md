DIAMOND PRICE PREDICTION
By John Beckstead

This is a term project for CIS 508, designed to showcase the end-to-end machine learning process.
The goal of this project, as states by the project guidelines, is to:
  1. Formulate a business problem as a machine learning task.
  2. Execute an end-to-end data mining and modeling process.
  3. Communicate your findings and business impact effectively.

diamonds.csv

This file is the dataset used in the model training. It is a csv file containing 10 characteristics of 53,490 diamonds. Each row corresponds to one unique diamond, and
each column corresponds to a characteristic of that diamonds. The characteristics are:
  carat (weight of a diamond. One carat is equal to 0.2 grams
  cut (the quality of the overall shaping of the finished diamond)
  color (diamond color is divided into 7 grades. D-F diamonds are considered colorless, while G-J have a very faint color)
  clarity (the amount of imperfections like cracks and mineral deposits, with 5 ordered levels: "Fair", "Good", "Very Good", "Premium", "Ideal")
  x, y, z, depth, table (various measures of a diamonds size, in millimeters)

  Diamond Price Prediction Notebook - v2.ipynb
  
  This is the python notebook file used to perform the machine learning portion of the project. To reproduce, simply import the notebook and the diamonds.csv file into
  a Databricks workspace, and run the file. This produces 3 runs each of 3 different machine learning algorithsm: Random Forest, SVR, and XGBoost. In my testing, the SVR runs took
  the longest, with the second SVR run averaging 35-36 minutes to complete each time the file was executed.

  app.py
  
  This is the python script used to generate the Streamlit app. This file can be run by opening the command terminal, navigating
  to the directory containing the app.py file and running the command: 'python -m streamlit run app.py'. This will open the streamlit app in
  your default browser. Alternatively, you can view the published version of the app [here](https://diamond-price-prediction-ribhe3u8camjkywknopkn2.streamlit.app/)
