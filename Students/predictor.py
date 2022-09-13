import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing


class Predictor:
    def __init__(self, name, model, args = None):
        """
        Constructor   
        :param name:  A name given to your predictor
        :param model: An instance of your ANN model class.
        :param parameters: An optional dictionary with parameters passed down to constructor.
        """
        self.name_ = name
        self.model_ = model
        #
        # You can add new member variables if you like.
        #
        return

    def get_name(self):
        """
        Return the name given to your predictor.   
        :return: name
        """
        return self.name_

    def get_model(self):
        """
         Return a reference to you model.
         :return: a model  
         """
        return self.model_

    # function that merges two datasets
    def merge_datasets(self, starting_df, df_to_merge, parameters):
      return starting_df.merge(df_to_merge, on=parameters)

    def predict(self, info_company, info_quarter, info_daily, current_stock_price):
        
        """
        Predict, based on the most recent information, the development of the stock-prices for companies 0-2.
        :param info_company: A list of information about each company
                             (market_segment.txt  records)
        :param info_quarter: A list of tuples, with the latest quarterly information for each of the market sectors.
                             (market_analysis.txt records)
        :param info_daily: A list of tuples, with the latest daily information about each company (0-2).
                             (info.txt  records)
        :param current_stock_price: A list of floats, with the with the current stock prices for companies 0-2.

        :return: A Python 3-tuple with your predictions: go-up (True), not (False) [company0, company1, company2]
        """

        # list of 3 tuples, 2 elements per tuple
        labels_info_company = ["company", "segment"]
        # print(info_company)

        # list of 2 tuples, 4 elements per tuple
        labels_info_quarter = ["segment", "year", "quarter", "trend"]
        # print(info_quarter)

        # info_daily -> list of 3 tuples, 11 elements per tuple
        labels_info_daily = ['company', 'year', 'day','quarter','expert1_prediction','expert2_prediction','sentiment_analysis','m1','m2','m3','m4']
        
        # current_stock_price -> after this operation: list of 3 tuples, 2 
        # elements per tuple
        labels_current_stock_price = ['company', 'stock_price']
        current_stock_price = [(0, current_stock_price[0]),(1, current_stock_price[1]),(2, current_stock_price[2])]
        # print(current_stock_price)

        # datasets creation, one per list
        df_info_company = pd.DataFrame(info_company, columns=labels_info_company)
        df_info_quarter = pd.DataFrame(info_quarter, columns=labels_info_quarter)
        df_info_daily = pd.DataFrame(info_daily, columns=labels_info_daily)
        df_current_stock_price = pd.DataFrame(current_stock_price, columns=labels_current_stock_price)

        # PREPROCESSING

        # merging of 4 datasets
        df = self.merge_datasets(df_current_stock_price, df_info_company, ['company'])
        df = self.merge_datasets(df, df_info_daily, ['company'])
        df = self.merge_datasets(df, df_info_quarter, ['segment', 'year', 'quarter'])

        df = df.sort_values(by ='company')

        # replace of "segment" literals values into numeric values
        df = df.replace(to_replace='BIO', value=0)
        df = df.replace(to_replace='IT', value=1)

        # normalization of values
        minmax_scaler = preprocessing.MinMaxScaler()
        df_final = minmax_scaler.fit_transform(df.values)
        df_new = pd.DataFrame(df_final, columns=['company', 'stock_price', 'segment', 	'year', 	'day' ,	'quarter',  		'expert1_prediction', 	'expert2_prediction' 	,'sentiment_analysis', 	'm1', 	'm2', 	'm3', 	'm4','trend', ])

        # empty data removal
        df.dropna(inplace = True) # Remove rows with NaN
        # ordering the dataframe like our training set
        df_new=df_new[['company', 'year', 	'day' ,	'quarter', 	'stock_price', 	'segment', 	'trend', 	'expert1_prediction', 	'expert2_prediction' 	,'sentiment_analysis', 	'm1', 	'm2', 	'm3', 	'm4']]

        # create tensor and DataLoader
        X_0 = torch.tensor(df_new.values, dtype=torch.float)
        test_ds = torch.utils.data.TensorDataset(X_0)
        test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False)
        
        # list that will contain our predictions
        companies = []

        model = self.get_model()
        with torch.no_grad():   # No need for keepnig track of necessary changes to the gradient.
          for data in test_dl:
            X = data[0]
            output = model(X.view(-1, 14))
            for idx, val in enumerate(output):
              companies.append(torch.argmax(val).item())

        return bool(companies[0]), bool(companies[1]), bool(companies[2])