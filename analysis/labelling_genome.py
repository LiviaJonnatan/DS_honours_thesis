# Import required libraries
#import warnings
#from pandas.errors import SettingWithCopyWarning
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class LabellingGenome:
    #warnings.simplefilter(action="ignore",category=SettingWithCopyWarning)
    #pd.options.mode.chained_assignment = None
    print("Labelling using genome data")
    merged= pd.read_csv("merge_complete.csv")
    genome_df = pd.read_csv("cleaned_genome_data.csv")
    genome_df.dt = pd.to_datetime(genome_df["dt"], errors='coerce')
    merged.dt = pd.to_datetime(merged["dt"], errors='coerce')
    merged["estrus"] = 0

    genome_df["key"] = 0
    for i in range(len(genome_df)):
        genome_df.key[i] = str(int(genome_df.id[i])) + "#" + str(genome_df.dt[i].date()) + "#" + str(genome_df.dt[i].hour)
  
    def genome_label(genome,df):
        for i in range(len(genome)):
            for j in range(len(df)):
                if ((genome.key[i] == df.key[j]) and (genome.trueestrus[i] == 1)):
                    final_dt = df.dt[j]+timedelta(hours=genome.afiduration[i])
                    for k in range(j, len(df)):
                        if (df.dt[k] > final_dt):
                            break
                        else:
                            df.estrus[k]=1
                    break
        return df
    
    merged_id = list(set(merged["AnimalID"]))
    merged_id.sort()

    print("Start loop")
    for i in range(len(merged_id)):
        print(merged_id[i])
        sub_df = (merged[merged.AnimalID==merged_id[i]])
        sub_df.reset_index(drop=True, inplace=True)
        new_df = genome_label(genome_df,sub_df)
        print("Saving to csv")
        filename = "gen_label_" + str(merged_id[i]) + ".csv"
        new_df.to_csv(filename, index=False)

    print("Done :)")
