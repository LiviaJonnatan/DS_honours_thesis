#Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CleaningGenomeData:
    print("Cleaning genome data")
    # Read excel file
    df = pd.read_excel("dataset/GenomeData_2017_2021.xlsx")

    df1 = df[["Cow", "EstrusDate", "TrueEstrus", "afithreshtime", "AFIDuration"]]
    df1.rename(columns={"Cow":"id"}, inplace=True)
    df1.rename(columns=str.lower, inplace=True)

    df1.afiduration.fillna(0, inplace=True)

    def hour_rounder(df):
        dt = pd.to_datetime(df["afithreshtime"], errors='coerce')
        rounded = [0 for i in range(len(dt))]
        for i in range(len(dt)):
            if (dt[i] != ""):
                if (dt[i].minute >= 30):
                    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
                    rounded[i] = (dt[i].replace(second=0, microsecond=0, minute=0, hour=dt[i].hour)+timedelta(hours=1))
                else:
                    rounded[i] = (dt[i].replace(second=0, microsecond=0, minute=0, hour=dt[i].hour))
        df["dt"] = rounded
        return df

    df1 = hour_rounder(df1)

    genome_df = df1
    year = [2020, 2021]
    for i in range(len(genome_df)):
        if (genome_df.dt[i].year not in year):
            genome_df = genome_df.drop(i)
        elif (genome_df.dt[i].year == 2020 and genome_df.dt[i].month < 12):
            genome_df = genome_df.drop(i)
        elif (genome_df.dt[i].year == 2020 and genome_df.dt[i].month == 12 and genome_df.dt[i].day < 20):
            genome_df = genome_df.drop(i)
    genome_df.reset_index(drop=True, inplace=True)
    
    genome_df.to_csv("cleaned_genome_data.csv", index=False)
