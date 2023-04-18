# Import required library
import pandas as pd

gen= pd.read_csv("all_gen.csv")
r_df = pd.read_csv("all_R.csv")

r_df.loc[r_df['est_new'] == "NORM", 'est_int'] = 8
r_df.loc[r_df['est_new'] == "EST", 'est_int'] = 10

perform = pd.DataFrame(columns=['diff', 'group'])
perform['diff'] = gen['estrus'] - r_df['est_int']

perform.loc[perform['diff'] == (-9), 'group'] = "TP"
perform.loc[perform['diff'] == (-7), 'group'] = "FN"
perform.loc[perform['diff'] == (-10), 'group'] = "FP"
perform.loc[perform['diff'] == (-8), 'group'] = "TN"

print("Saving performance df")
perform.to_csv("perform_result.csv", index=False)

print(perform.groupby(by="group").count())

TP_count = perform[perform["group"] == 'TP'].shape[0]
FP_count = perform[perform["group"] == 'FP'].shape[0]
FN_count = perform[perform["group"] == 'FN'].shape[0]
TN_count = perform[perform["group"] == 'TN'].shape[0]

print("Calculating accuracy")
accu = ((TP_count + TN_count) / (TP_count + TN_count + FP_count + FN_count)) * 100
print("Calculating specificity")
spec = (TN_count / (TN_count + FP_count)) * 100
print("Calculating precision")
prec = (TP_count / (TP_count + FP_count)) * 100
print("Calculating recall")
recall = (TP_count / (TP_count + FN_count)) * 100
print("Calculating F1")
f1 = (2 * prec * recall / (prec + recall)) * 100

print("Accuracy: ", accu, "%")
print("Specificity: ", spec, "%")
print("Precision: ", prec, "%")
print("Recall: ", recall, "%")
print("F1 Score: ", f1, "%")
print("Done")
