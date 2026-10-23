# Machine learning model that predicts the scores students get using the number of hours they study per week
# The user will input their number of hours studying and a the model will predict their grade
# Data exploration and cleaning will have been done in a Jupyter notebook and will not appear here

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

def main():
    # Random dataset I found with a tiny number of samples, for more accurate predictions, find a better dataset
    df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
    model, rmse = train_model(df)

    while True:
        hours = input("Number of study hours per week: ")
        try:
            hours = float(hours)
            break
        except:
            print("Enter a number, you doofus! ")

    score = min(model.predict(pd.DataFrame([[hours]], columns=model.feature_names_in_))[0], 100)
    print(f"I predict your score is: {int(score)} +- {int(rmse)}")

    if score < 20:
        print("Start praying")
    elif score < 50:
        print("Go and study")
    else:
        print("What a nerd")


def train_model(df):
    # Splitting data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    y_train = df_train.pop("Scores")
    y_test = df_test.pop("Scores")

    # Training model
    model = LinearRegression()
    model.fit(df_train, y_train)

    # Evaluating model
    pred = model.predict(df_test)
    rmse = root_mean_squared_error(y_test, pred)

    return model, rmse

if __name__ == "__main__":
    main()