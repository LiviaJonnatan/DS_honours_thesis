# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Merging:
    print("Merging 15 minutes data:")
    # Read excel file
    genome_df = pd.read_csv("cleaned_genome_data.csv")
    genome_id = genome_df.sort_values(by="id", ascending = True).drop_duplicates(subset = ["id"]).id.tolist()
    
    def merge_min_data(data, genome_id):
        df = pd.DataFrame()
        for i in range(len(data)):
            new_df = pd.read_excel(data[i])
            new_df = new_df[["Time", "AnimalID", "Step"]]
            for k in range(len(new_df)):
                if new_df.AnimalID[k] not in genome_id:
                    new_df = new_df.drop(k)
            df = df.append(new_df)
        df.reset_index(drop=True, inplace=True)
        dt = pd.to_datetime(df["Time"], errors='coerce')
        rounded = [0 for i in range(len(dt))]
        for j in range(len(dt)):
            if (dt[j].second >= 30):
                # Rounds to nearest minute by adding a timedelta minute if second >= 30
                rounded[j] = (dt[j].replace(second=0, microsecond=0, minute=dt[j].minute, hour=dt[j].hour)+timedelta(minutes=1))
            else:
                rounded[j] = (dt[j].replace(second=0, microsecond=0, minute=dt[j].minute, hour=dt[j].hour))
        df["dt"] = rounded
        df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
        df.reset_index(drop=True, inplace=True)
        return df
    
    fifteen_min_data = ["dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20201220-20201226.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20201227-20210102.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210103-20210109.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210110-20210116.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210117-20210123.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210124-20210130.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210131-20210206.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210207-20210213.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210214-20210220.xlsx",
                        #"dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210221-20210227.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210221-20210320.xlsx",
                        #"dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210228-20210306.xlsx",
                        #"dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210307-20210313.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210321-20210403.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210404-20210410.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210411-20210424.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210425-20210509.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210510-20210523.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210524-20210606.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210601-20210708.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210709-20210731.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210801-20210831.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20210901-20210930.xlsx",
                        "dataset/15_min_steps_data/AfiAct2_Steps and Activity_15 minutes data_20211001-20211030.xlsx"]

    merged_df = merge_min_data(fifteen_min_data, genome_id)
    merged_df.to_csv("merged.csv")

    cows_id = merged_df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    print("Count cows")
    print(len(cows_id))

    print("Done :)")