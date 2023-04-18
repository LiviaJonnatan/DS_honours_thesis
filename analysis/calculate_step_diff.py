# Import required library
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# read dataset
gen = pd.read_csv("/dataset/all_gen.csv")

pd.set_option("mode.chained_assignment", None)

def diff_step(df):
    diffs = []
    # sort by id and dt collected
    df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
    # get only unique animal id
    animalids = df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    # for each unique id
    for i in range(len(animalids)):
        # get the subset with that id
        temp = df[df.AnimalID == animalids[i]]
        # get a list of all the steps
        step_count = temp["Step"].tolist()
        # calculate the difference between each row
        diff = temp["Step"].diff().tolist()
        # for each calculated difference
        for j in range(len(diff)):
            # if the difference is a large negative number, the sensor was resetted
            if diff[j] < -1000:
                # change to current step count + threshold - previous step count
                diffs.append((step_count[j]+65535)-step_count[j-1])
            else:
                diffs.append(diff[j])
    # add step difference into dataframe
    df["step_diff"] = diffs
    return df

def diff_step_30min(df):
    diffs = []
    # sort by id and dt collected
    df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
    # get only unique animal id
    animalids = df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    # for each unique id
    for i in range(len(animalids)):
        # get the subset with that id
        temp = df[df.AnimalID == animalids[i]]
        # get a list of all the steps
        step_count = temp["Step"].tolist()
        # calculate the difference between each row
        diff = temp["Step"].diff(2).tolist()
        # for each calculated difference
        for j in range(len(diff)):
            # if the difference is a large negative number, the sensor was resetted
            if diff[j] < -1000:
                # change to current step count + threshold - previous step count
                diffs.append((step_count[j]+65535)-step_count[j-1])
            else:
                diffs.append(diff[j])
    # add step difference into dataframe
    df["step_diff_30min"] = diffs
    return df

def diff_step_hr(df):
    diffs = []
    # sort by id and dt collected
    df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
    # get only unique animal id
    animalids = df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    # for each unique id
    for i in range(len(animalids)):
        # get the subset with that id
        temp = df[df.AnimalID == animalids[i]]
        # get a list of all the steps
        step_count = temp["Step"].tolist()
        # calculate the difference between each row
        diff = temp["Step"].diff(4).tolist()
        # for each calculated difference
        for j in range(len(diff)):
            # if the difference is a large negative number, the sensor was resetted
            if diff[j] < -1000:
                # change to current step count + threshold - previous step count
                diffs.append((step_count[j]+65535)-step_count[j-1])
            else:
                diffs.append(diff[j])
    # add step difference into dataframe
    df["step_diff_hr"] = diffs
    return df

def diff_step_one_half(df):
    diffs = []
    # sort by id and dt collected
    df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
    # get only unique animal id
    animalids = df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    # for each unique id
    for i in range(len(animalids)):
        # get the subset with that id
        temp = df[df.AnimalID == animalids[i]]
        # get a list of all the steps
        step_count = temp["Step"].tolist()
        # calculate the difference between each row
        diff = temp["Step"].diff(6).tolist()
        # for each calculated difference
        for j in range(len(diff)):
            # if the difference is a large negative number, the sensor was resetted
            if diff[j] < -1000:
                # change to current step count + threshold - previous step count
                diffs.append((step_count[j]+65535)-step_count[j-1])
            else:
                diffs.append(diff[j])
    # add step difference into dataframe
    df["step_diff_one_half"] = diffs
    return df

def diff_step_3hr(df):
    diffs = []
    # sort by id and dt collected
    df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
    # get only unique animal id
    animalids = df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    # for each unique id
    for i in range(len(animalids)):
        # get the subset with that id
        temp = df[df.AnimalID == animalids[i]]
        # get a list of all the steps
        step_count = temp["Step"].tolist()
        # calculate the difference between each row
        diff = temp["Step"].diff(12).tolist()
        # for each calculated difference
        for j in range(len(diff)):
            # if the difference is a large negative number, the sensor was resetted
            if diff[j] < -1000:
                # change to current step count + threshold - previous step count
                diffs.append((step_count[j]+65535)-step_count[j-1])
            else:
                diffs.append(diff[j])
    # add step difference into dataframe
    df["step_diff_3hr"] = diffs
    return df

def diff_step_6hr(df):
    diffs = []
    # sort by id and dt collected
    df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
    # get only unique animal id
    animalids = df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    # for each unique id
    for i in range(len(animalids)):
        # get the subset with that id
        temp = df[df.AnimalID == animalids[i]]
        # get a list of all the steps
        step_count = temp["Step"].tolist()
        # calculate the difference between each row
        diff = temp["Step"].diff(24).tolist()
        # for each calculated difference
        for j in range(len(diff)):
            # if the difference is a large negative number, the sensor was resetted
            if diff[j] < -1000:
                # change to current step count + threshold - previous step count
                diffs.append((step_count[j]+65535)-step_count[j-1])
            else:
                diffs.append(diff[j])
    # add step difference into dataframe
    df["step_diff_6hr"] = diffs
    return df

def diff_step_12hr(df):
    diffs = []
    # sort by id and dt collected
    df = df.sort_values(by=["AnimalID", "dt"], ascending = [True, True])
    # get only unique animal id
    animalids = df.drop_duplicates(subset = ["AnimalID"]).AnimalID.tolist()
    # for each unique id
    for i in range(len(animalids)):
        # get the subset with that id
        temp = df[df.AnimalID == animalids[i]]
        # get a list of all the steps
        step_count = temp["Step"].tolist()
        # calculate the difference between each row
        diff = temp["Step"].diff(48).tolist()
        # for each calculated difference
        for j in range(len(diff)):
            # if the difference is a large negative number, the sensor was resetted
            if diff[j] < -1000:
                # change to current step count + threshold - previous step count
                diffs.append((step_count[j]+65535)-step_count[j-1])
            else:
                diffs.append(diff[j])
    # add step difference into dataframe
    df["step_diff_12hr"] = diffs
    return df

gen_df = diff_step(gen)
gen_df = diff_step_30min(gen_df)
gen_df = diff_step_hr(gen_df)
gen_df = diff_step_one_half(gen_df)
gen_df = diff_step_3hr(gen_df)
gen_df = diff_step_6hr(gen_df)
gen_df = diff_step_12hr(gen_df)

gen_df2 = gen_df
gen_df2["dt"] = pd.to_datetime(gen_df2["dt"], errors='coerce')
gen_df2["minute"] = gen_df2["dt"].dt.minute
gen_df2["hour"] = gen_df2["dt"].dt.hour
gen_df2["month"] = gen_df2["dt"].dt.month

# Hourly average step difference with 30 min. interval by cow
temp = gen_df2.groupby(["AnimalID", "hour"]).agg(step_diff_30min_mean_by_cow = ("step_diff_30min", "mean"))
temp = temp.reset_index()
ids = temp["AnimalID"].tolist()
hr = temp.hour.tolist()
diff_means_by_cow = temp.step_diff_30min_mean_by_cow.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_30min_mean_by_cow"] = 0
for i in range(len(ids)):
    gen_df2.step_diff_30min_mean_by_cow[(gen_df2.AnimalID == ids[i]) & (gen_df2.hour == hr[i])] = diff_means_by_cow[i]

# Hourly average step difference with 1hr interval by cow
temp = gen_df2.groupby(["AnimalID", "hour"]).agg(step_diff_hr_mean_by_cow = ("step_diff_hr", "mean"))
temp = temp.reset_index()
ids = temp["AnimalID"].tolist()
hr = temp.hour.tolist()
diff_means_by_cow = temp.step_diff_hr_mean_by_cow.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_hr_mean_by_cow"] = 0
for i in range(len(ids)):
    gen_df2.step_diff_hr_mean_by_cow[(gen_df2.AnimalID == ids[i]) & (gen_df2.hour == hr[i])] = diff_means_by_cow[i]

# Hourly average step difference with 1.5 hr interval by cow
temp = gen_df2.groupby(["AnimalID", "hour"]).agg(step_diff_one_half_mean_by_cow = ("step_diff_one_half", "mean"))
temp = temp.reset_index()
ids = temp["AnimalID"].tolist()
hr = temp.hour.tolist()
diff_means_by_cow = temp.step_diff_one_half_mean_by_cow.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_one_half_mean_by_cow"] = 0
for i in range(len(ids)):
    gen_df2.step_diff_one_half_mean_by_cow[(gen_df2.AnimalID == ids[i]) & (gen_df2.hour == hr[i])] = diff_means_by_cow[i]

# Hourly average step difference with 3 hr interval by cow
temp = gen_df2.groupby(["AnimalID", "hour"]).agg(step_diff_3hr_mean_by_cow = ("step_diff_3hr", "mean"))
temp = temp.reset_index()
ids = temp["AnimalID"].tolist()
hr = temp.hour.tolist()
diff_means_by_cow = temp.step_diff_3hr_mean_by_cow.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_3hr_mean_by_cow"] = 0
for i in range(len(ids)):
    gen_df2.step_diff_3hr_mean_by_cow[(gen_df2.AnimalID == ids[i]) & (gen_df2.hour == hr[i])] = diff_means_by_cow[i]

# Hourly average step difference with 6 hr interval by cow
temp = gen_df2.groupby(["AnimalID", "hour"]).agg(step_diff_6hr_mean_by_cow = ("step_diff_6hr", "mean"))
temp = temp.reset_index()
ids = temp["AnimalID"].tolist()
hr = temp.hour.tolist()
diff_means_by_cow = temp.step_diff_6hr_mean_by_cow.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_6hr_mean_by_cow"] = 0
for i in range(len(ids)):
    gen_df2.step_diff_6hr_mean_by_cow[(gen_df2.AnimalID == ids[i]) & (gen_df2.hour == hr[i])] = diff_means_by_cow[i]

# Hourly average step difference with 12 hr interval by cow
temp = gen_df2.groupby(["AnimalID", "hour"]).agg(step_diff_12hr_mean_by_cow = ("step_diff_12hr", "mean"))
temp = temp.reset_index()
ids = temp["AnimalID"].tolist()
hr = temp.hour.tolist()
diff_means_by_cow = temp.step_diff_12hr_mean_by_cow.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_12hr_mean_by_cow"] = 0
for i in range(len(ids)):
    gen_df2.step_diff_12hr_mean_by_cow[(gen_df2.AnimalID == ids[i]) & (gen_df2.hour == hr[i])] = diff_means_by_cow[i]

# Hourly average step difference with 30 min. interval over all cows
temp = gen_df2.groupby(["hour"]).agg(step_diff_30min_mean_overall = ("step_diff_30min", "mean"))
temp = temp.reset_index()
hr = temp.hour.tolist()
diff_means_overall = temp.step_diff_30min_mean_overall.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_30min_mean_overall"] = 0
for i in range(len(hr)):
    gen_df2.step_diff_30min_mean_overall[(gen_df2.hour == hr[i])] = diff_means_overall[i]

# Hourly average step difference with 1 hr interval over all cows
temp = gen_df2.groupby(["hour"]).agg(step_diff_hr_mean_overall = ("step_diff_hr", "mean"))
temp = temp.reset_index()
hr = temp.hour.tolist()
diff_means_overall = temp.step_diff_hr_mean_overall.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_hr_mean_overall"] = 0
for i in range(len(hr)):
    gen_df2.step_diff_hr_mean_overall[(gen_df2.hour == hr[i])] = diff_means_overall[i]

# Hourly average step difference with 1.5 hr interval over all cows
temp = gen_df2.groupby(["hour"]).agg(step_diff_one_half_mean_overall = ("step_diff_one_half", "mean"))
temp = temp.reset_index()
hr = temp.hour.tolist()
diff_means_overall = temp.step_diff_one_half_mean_overall.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_one_half_mean_overall"] = 0
for i in range(len(hr)):
    gen_df2.step_diff_one_half_mean_overall[(gen_df2.hour == hr[i])] = diff_means_overall[i]

# Hourly average step difference with 3 hr interval over all cows
temp = gen_df2.groupby(["hour"]).agg(step_diff_3hr_mean_overall = ("step_diff_3hr", "mean"))
temp = temp.reset_index()
hr = temp.hour.tolist()
diff_means_overall = temp.step_diff_3hr_mean_overall.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_3hr_mean_overall"] = 0
for i in range(len(hr)):
    gen_df2.step_diff_3hr_mean_overall[(gen_df2.hour == hr[i])] = diff_means_overall[i]

# Hourly average step difference with 6 hr interval over all cows
temp = gen_df2.groupby(["hour"]).agg(step_diff_6hr_mean_overall = ("step_diff_6hr", "mean"))
temp = temp.reset_index()
hr = temp.hour.tolist()
diff_means_overall = temp.step_diff_6hr_mean_overall.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_6hr_mean_overall"] = 0
for i in range(len(hr)):
    gen_df2.step_diff_6hr_mean_overall[(gen_df2.hour == hr[i])] = diff_means_overall[i]

# Hourly average step difference with 12 hr interval over all cows
temp = gen_df2.groupby(["hour"]).agg(step_diff_12hr_mean_overall = ("step_diff_12hr", "mean"))
temp = temp.reset_index()
hr = temp.hour.tolist()
diff_means_overall = temp.step_diff_12hr_mean_overall.tolist()
# Fill calculated mean into original dataframe
gen_df2["step_diff_12hr_mean_overall"] = 0
for i in range(len(hr)):
    gen_df2.step_diff_12hr_mean_overall[(gen_df2.hour == hr[i])] = diff_means_overall[i]

gen_df2.to_csv("gen_mean_step_diff.csv",index=False)

print("Done :)")

