import pandas as pd

procs = pd.read_csv("/Users/rm/Downloads/MIMIC IV SP/d_icd_procedures.csv/d_icd_procedures.csv")

procs = procs[['icd_code', 'icd_version', 'long_title']]
procs_uniq = procs.drop_duplicates(keep=False)
procs_uniq.to_csv("procs.csv", index=False)