#Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class LabellingR:
    print("Labelling using R ways")
    merged_df = pd.read_csv("merge_complete.csv")
    merged_df.dt = pd.to_datetime(merged_df["dt"], errors='coerce')
    merged_df["minute"] = merged_df["dt"].dt.minute
    merged_df["hour"] = merged_df["dt"].dt.hour
    merged_df["month"] = merged_df["dt"].dt.month

    def r_label(df):
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
                    # if the difference is a large negative number, the sensor being reset
                    if diff[j] < -1000:
                        # change to current step count + threshold - previous step count
                        diffs.append((step_count[j]+65535)-step_count[j-1])
                    else:
                        diffs.append(diff[j])
            # add step difference into dataframe
            df["step_diff"] = diffs
            return df

        # Method to detect estrus based on duration
        def est_detection(x):
            if x.duration >= 4:
                return "EST"
            else:
                return "NORM"

        def est_detection_new(x):
            if x.duration_new >= 4:
                return "EST"
            else:
                return "NORM"

        df = diff_step(df)

        temp = df.groupby(["AnimalID", "minute"]).agg(step_mean = ("step_diff", "mean"))
        temp = temp.reset_index()

        ids = temp["AnimalID"].tolist()
        minute = temp.minute.tolist()
        means = temp.step_mean.tolist()

        # Fill calculated mean into original dataframe
        df["step_mean"] = 0
        for i in range(len(ids)):
            df.step_mean[(df.AnimalID == ids[i]) & (df.minute == minute[i])] = means[i]

        df["step_diff_imputed"] = df["step_diff"]

        # Fill step_diff_imputed with step_mean value for step_diff == na
        df.step_diff_imputed[(df.step_diff.isna())]= df["step_mean"]

        # Roll_mean is the 5 window rolling mean of step_diff_imputed, group by id, hour
        df["roll_mean"] = df.groupby(["AnimalID","hour"])["step_diff_imputed"].apply(lambda x: x.rolling(5).mean())
        df["perc"] = 100*(df["step_diff_imputed"]/df["roll_mean"])

        # Flag perc >= 180
        df["c8"] = df.apply(lambda x: 1 if x["perc"] >= 180 else 0, axis=1)

        # run length encoding
        temp = df.c8.tolist()
        ids = df["AnimalID"].tolist()

        rle_encode = []
        rle_ids = []
        rle_id = 1
        count = 0
        prev_c8 = 0
        prev_id = ids[0]
        #for value in c8
        for i in range(len(temp)):
            #if value is equal to the previous one increase count
            if ((temp[i] == prev_c8) & (ids[i] == prev_id)):
                count += 1
            else:
                #if not equal, reset
                count = 1
                if (ids[i] == prev_id):
                    rle_id += 1
                else:
                    rle_id = 1
            prev_c8 = temp[i]
            prev_id = ids[i]
            rle_ids.append(rle_id)
            rle_encode.append(count)

        df["rel1"] = rle_encode
        df["group"] = rle_ids
        
        temp = df.groupby(["AnimalID","group","hour"]).agg(duration = ("c8", "sum")).reset_index()

        ids = temp["AnimalID"].tolist()
        groups = temp.group.tolist()
        hrs = temp.hour.tolist()
        dur = temp.duration.tolist()

        # Fill duration into full dataframe
        df["duration"] = 0
        for i in range(len(ids)):
            df.duration[(df.AnimalID == ids[i]) & (df.group == groups[i]) & (df.hour == hrs[i])] = dur[i]

        # Call estrus detection method for each row
        df["est"] = df.apply(lambda x: est_detection(x), axis=1)

        df["c8_new"] = np.where(df['duration'] >= 4, 1, 0)

        # run length encoding
        temp = df.c8_new.tolist()
        ids = df["AnimalID"].tolist()

        rle_encode2 = []
        rle_group_id = []
        rle_id = 1
        count = 0
        prev_c8 = 0
        prev_id = ids[0]
        #for value in c8
        for i in range(len(temp)):
            #if value is equal to the previous one increase count
            if ((temp[i] == prev_c8) & (ids[i] == prev_id)):
                count += 1
            else:
                #if not equal, reset
                count = 1
                if (ids[i] == prev_id):
                    rle_id += 1
                else:
                    rle_id = 1
            prev_c8 = temp[i]
            prev_id = ids[i]
            rle_group_id.append(rle_id)
            rle_encode2.append(count)

        df["rle_new"] = rle_encode2
        df["group_new"] = rle_group_id

        durations = df.group_new.tolist()
        c8_vals = df.c8_new.tolist()
        count_groups = {}
        count = 0
        current_group = 1
        vals = []

        #increase estrus detection to include if values above and below a 0 are 1
        #this counts those
        for i in range(len(durations)):
            if current_group != durations[i]:
                vals.append({"count": count})
                vals.append({"c8_val": c8_vals[i-1]})
                count_groups[current_group] = vals
                current_group = durations[i]
                vals = []
                count = 0
            else:
                count += 1

        # if there is 4 hours of zeros but there are hours of 1 before and after, it should be considered as an event 
        events = []
        keys = list(count_groups.keys())
        values = list(count_groups.values())

        for i in range(len(keys)):
            count = values[i][0]["count"]
            if count <= 4 and values[i][1]["c8_val"] == 0 and i > 0 and i < len(keys)-1:
                if values[i-1][0]["count"] >= 1 and values[i-1][1]["c8_val"] == 1 and values[i+1][0]["count"] >= 1 and values[i+1][1]["c8_val"] == 1:
                    events.append(keys[i])

        for i in range(len(durations)):
            if durations[i] in events:
                c8_vals[i] == 1
        df["flag_new"] = c8_vals

        temp = df.groupby(["AnimalID","group_new","hour"]).agg(duration_new = ("flag_new", "sum")).reset_index()

        ids = temp["AnimalID"].tolist()
        groups = temp.group_new.tolist()
        hrs = temp.hour.tolist()
        dur = temp.duration_new.tolist()

        # Fill duration into full dataframe
        df["duration_new"] = 0
        for i in range(len(ids)):
            df.duration_new[(df.AnimalID == ids[i]) & (df.group_new == groups[i]) & (df.hour == hrs[i])] = dur[i]

        df["est_new"] = df.apply(lambda x: est_detection_new(x), axis=1)
        return df
    

    merged_id = list(set(merged_df["AnimalID"]))
    merged_id.sort()

    for i in range(len(merged_id)):
        df = merged_df[merged_df.AnimalID==merged_id[i]]
        df.reset_index(drop=True, inplace=True)
        new_df = r_label(df)
        filename = "r_label_" + str(merged_id[i]) + ".csv"
        new_df.to_csv(filename, index=False)
    
    print("Done :)")
