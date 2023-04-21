# Analysis
This directory has Python scripts to:
- data cleaning
- data merging
  ([complete and merged data](https://ubcca-my.sharepoint.com/:x:/g/personal/ramon_lawrence_ubc_ca/EVlzmtK_lwxDu2eAxZYz7m8BcYvZOeW8XBjcVKqUKDy0ig?e=6mmpRd))
- data labelling (result based on [sensor data](https://ubcca-my.sharepoint.com/:x:/g/personal/ramon_lawrence_ubc_ca/ET_fUww-dZFOhhGSWkj6j8EBhy0SnBYNT3FfkQEPO8Q0cg?e=85pcqo) and [R algorithm](https://ubcca-my.sharepoint.com/:x:/g/personal/ramon_lawrence_ubc_ca/ET66Wrn7qEVMv1cWQJev0UsByvyBp9ktxXFyCaiuQRK9Ig?e=UxSioL))
- compare result from previouly existing R algorithm and sensor data
- calculate step difference ([final CSV file](https://ubcca-my.sharepoint.com/:x:/g/personal/ramon_lawrence_ubc_ca/ERiZ10VBu-dBifB2QkEx1W4BFJVrlMUV-xKKRLRliMv6zg?e=5Qjnc1) for machine learning)
- supervised machine learning
  - classification tree
  - naive Bayes
  - K-nearest neighbors
  - logistic regression
  - linear discriminant analysis
  - support vector machines (linear, polynomial, radial basis function)
- exploring and plotting bovine steps
- principal component analysis, followed by mentioned supervised learning algorithms

P.S. "genome" and "sensor" are used interchangably, but refer to the same thing (sensor data)

## Set up
Python libraries that must be installed:
- pandas
- numpy
- datetime
- matplotlib
- scikit-learn

Install in Terminal by using `pip3 install requirements.txt`.

Run all Python scripts (.py) in Terminal by using this command: `python3 filename.py`.

For .ipnyb file, open it using Jupyter Notebook.

## Flow
1. Run `cleaning_genome_data.py` -> return `cleaned_genome_data.csv`
2. Run `merging.py` -> return `merge_complete.csv`
3. Run `labelling_R.py` -> return CSV files for individual cow (`r_label_(animalID).csv`)
4. Run `labelling_genome.py` -> return CSV files for individual cow (`gen_label_(animalID).csv`)
5. Run `df_complete.py` -> return `all_gen.csv` and `all_R.csv`
6. Run `compare_R_sensor.py` -> return `result_compare_R_sensor.txt`
7. Run `calculate_step_diff.py` -> return `gen_mean_step_diff.csv`
8. Run `ML_mean_step_diff_cow.py` -> return `output_ML_mean_step_diff_cow.txt`
9. Run `ML_mean_step_diff_overall.py` -> return `output_ML_mean_step_diff_overall.txt`
10. Run `ML_step_diff.py` -> return `output_ML_step_diff.txt`
11. Run `plot_steps.ipnyb`
12. Run `explore_PCA.ipnyb`
13. Run `ML_PCA.py`
