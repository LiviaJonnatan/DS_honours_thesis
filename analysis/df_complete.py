# Import required library
import pandas as pd

merged= pd.read_csv("merge_complete.csv")
merged_id = list(set(merged["AnimalID"]))
merged_id.sort()

gen = pd.DataFrame()
r_df = pd.DataFrame()

print("Start loop")
for i in range(len(merged_id)):
    gen_filename = "gen_label_" + str(merged_id[i]) + ".csv"
    gen_data = pd.read_csv(gen_filename)
    gen = gen.append(gen_data)
    r_filename = "r_label_" + str(merged_id[i]) + ".csv"
    r_data = pd.read_csv(r_filename)
    r_df = r_df.append(r_data)

print("Done loop")
gen.reset_index(drop=True, inplace=True)
r_df.reset_index(drop=True, inplace=True)

r_df = r_df[["AnimalID", "Step", "step_mean", "step_diff_imputed", "dt", "duration_new", "est_new"]]
r_df.rename(columns={"Step":"step"}, inplace=True)

print("Saving genome")
gen.to_csv("all_gen.csv", index=False)
print("Complete genome df done")
print("Saving R")
r_df.to_csv("all_R.csv", index=False)
print("Complete R df done")
