def wrangle(filepath):
    df= pd.read_csv(filepath)
    df["may_flood"]= [1 if val >0.5 else 0 for val in df["FloodProbability"]]

    return df

def make_predictions(data_filepath, model_filepath):
    #Wrangle csv file
    X_test= wrangle(data_filepath)

    #Load our model
    with open(model_filepath, "rb") as f:
        model= pickle.load(f)


    #Generate Predictions
    y_test_pred= model.predict(X_test)
    #Putting predictions into series with name "will_flood"
    y_test_pred= pd.Series(y_test_pred, index=X_test.index, name="Will_Flood")

    return y_test_pred