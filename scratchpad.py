import pandas as pd

labe = pd.read_csv("/Users/rm/Downloads/MIMIC IV SP/labevents.csv/labevents.csv")

labe = labe.sample(n=10000)

labe.to_csv("labe_sample.csv", index=False)