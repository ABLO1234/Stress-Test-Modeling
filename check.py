import pandas as pd



if __name__ == "__main__":
    df = pd.read_csv("data_bancaire.csv")

    print(df.shape)
    print(len(df))