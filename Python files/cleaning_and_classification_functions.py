import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

def cleaning_function(X_train, y_train, X_test, one_hot_encoding=False):

    # Creating a matrix with X and y merged
    X_with_label = pd.merge(X_train, y_train, how='left', on="Trader")

    # Removing duplicates
    X_with_label.drop_duplicates(inplace=True)
    X_test.drop_duplicates(inplace=True)

    # Converting date and share to integers to do one-hot encoding
    X_with_label["Share"] = pd.to_numeric(X_with_label["Share"].str[5::], downcast='integer')
    X_with_label["Day"] = pd.to_numeric(X_with_label["Day"].str[5::], downcast='integer')

    X_test["Share"] = pd.to_numeric(X_test["Share"].str[5::], downcast='integer')
    X_test["Day"] = pd.to_numeric(X_test["Day"].str[5::], downcast='integer')    

    # TODO One hot encoding

    if one_hot_encoding == True:
        one_hot_encoder = OneHotEncoder()
        one_hot_cols = ["Day", "Share"]
        
        preprocessor = make_column_transformer(
            (one_hot_encoder, one_hot_cols),
            remainder='passthrough'
        )
        preprocessor.fit_transform(X_with_label)
        preprocessor.transform(X_test)


    # Treatment of OTR, OMR and OCR
    ## Putting NAs to 0 in OTR, OCR and OMR
    X_with_label[["OTR", "OMR", "OCR"]] = X_with_label[["OTR", "OMR", "OCR"]].fillna(value=0)
    X_test[["OTR", "OMR", "OCR"]] = X_test[["OTR", "OMR", "OCR"]].fillna(value=0)

    ## Total column
    X_with_label["Total OR"] = X_with_label["OTR"] + X_with_label["OMR"] + X_with_label["OCR"]
    X_test["Total OR"] = X_test["OTR"] + X_test["OMR"] + X_test["OCR"]

    # Replacing columns OTR, OMR and OCR to a new one making more sense with proportions
    X_with_label["OTR_new"] = X_with_label["OTR"] / X_with_label["Total OR"]
    X_with_label["OMR_new"] = X_with_label["OMR"] / X_with_label["Total OR"]
    X_with_label["OCR_new"] = X_with_label["OCR"] / X_with_label["Total OR"]
    X_test["OTR_new"] = X_test["OTR"] / X_test["Total OR"]
    X_test["OMR_new"] = X_test["OMR"] / X_test["Total OR"]
    X_test["OCR_new"] = X_test["OCR"] / X_test["Total OR"]

    X_with_label.drop(columns=["OTR", "OMR", "OCR"], inplace=True)
    X_test.drop(columns=["OTR", "OMR", "OCR"], inplace=True)
    
    # Deleting columns that have too much correlation with some others
    X_with_label.drop(columns=[
        "10_p_time_two_events", "25_p_time_two_events", 
        "med_time_two_events", "75_p_time_two_events", 
        "90_p_time_two_events", "min_lifetime_cancel", 
        "10_p_lifetime_cancel", "med_lifetime_cancel", 
        "25_p_lifetime_cancel", "75_p_lifetime_cancel",
        "90_p_lifetime_cancel", "max_lifetime_cancel"
        ], inplace=True)
    X_test.drop(columns=[
        "10_p_time_two_events", "25_p_time_two_events", 
        "med_time_two_events", "75_p_time_two_events", 
        "90_p_time_two_events", "min_lifetime_cancel", 
        "10_p_lifetime_cancel", "med_lifetime_cancel", 
        "25_p_lifetime_cancel", "75_p_lifetime_cancel",
        "90_p_lifetime_cancel", "max_lifetime_cancel"
        ], inplace=True)

    # Removing other NAs for now
    # X_with_label.dropna(axis="columns", inplace=True)
    # X_test.dropna(axis="columns", inplace=True)
    
    # Preparing output matrixes
    y_train_reshaped = X_with_label['type']
    X_anonymized = X_with_label.drop(columns=['Trader', 'type'])

    return X_anonymized, y_train_reshaped, X_test, X_with_label


def predictions_matrix(traders, y_pred):

    res = pd.DataFrame(traders)
    res['pred'] = y_pred
    res['count'] = 1

    predictions = res.groupby(['Trader', 'pred']).count() / res.groupby(['Trader']).count()
    predictions = predictions.unstack(level=1).drop(columns=['pred']).fillna(0)
    predictions.columns = predictions.columns.get_level_values(1)
    
    return predictions


def final_classification(predictions):

    predictions.reset_index(inplace=True)
    predictions['type'] = 'NON HFT'
    
    for i in range(len(predictions)):
        if predictions.iloc[i]['HFT'] >= 0.85:
            predictions.at[i, 'type'] = 'HFT'
        elif predictions.iloc[i]['MIX'] > 0.5:
            predictions.at[i, 'type'] = 'MIX'
    predictions.drop(columns=['HFT','MIX','NON HFT'], inplace=True)
    
    return predictions