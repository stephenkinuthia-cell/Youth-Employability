import pandas as pd
import os
base = r'c:\Users\kinut\Youth-Employability'
files = [
    'classification_model_results.csv',
    'regression_model_results.csv',
    'global_graduate_employability_index.csv',
    'model_ready_data.csv',
    'employability-ai-system\\data\\processed\\final_features.csv'
]
for path in files:
    p = os.path.join(base, path)
    print('FILE:', path, 'EXISTS', os.path.exists(p))
    if os.path.exists(p):
        if path.endswith('.csv'):
            df = pd.read_csv(p, nrows=5)
            print('shape (sample)', df.shape)
            print('columns', list(df.columns)[:15])
            if path == 'global_graduate_employability_index.csv':
                print('first rows sample:')
                print(df.to_string(index=False))
        if path == 'global_graduate_employability_index.csv' or path == 'model_ready_data.csv' or path == 'employability-ai-system\\data\\processed\\final_features.csv':
            with open(p, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print('line count', lines)
    print('---')
