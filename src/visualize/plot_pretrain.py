import pandas as pd
import re

olmo_7b_file = '../../pretrain_logs/wandb_export_2025-02-12T17_48_01.405+05_30.csv'
olmo_1b_file = '../../pretrain_logs/wandb_export_2025-02-12T17_47_33.373+05_30.csv'

olmo_7b = pd.read_csv(olmo_7b_file)
olmo_1b = pd.read_csv(olmo_1b_file)


merge_patterns_7b = re.compile(r'^OLMo-7B-run-\d{3} - eval/c4_100_domains-validation/Perplexity$')
delete_patterns_7b = re.compile(r'^OLMo-7B-run-\d{3} - eval/c4_100_domains-validation/Perplexity__M..$')


merge_patterns_1b = re.compile(r'^OLMo-1B-run-\d{3} - eval/v2-small-c4_100_domains-validation/Perplexity')
delete_patterns_1b = re.compile(r'^OLMo-1B-run-\d{3} - eval/v2-small-c4_100_domains-validation/Perplexity__M..$')

def clean_df(df, merge_patterns, delete_patterns, new_col):

    # Extract columns matching the pattern
    merge_cols = [col for col in df.columns if merge_patterns.match(col)]
    delete_cols = [col for col in df.columns if delete_patterns.match(col)]

    df[new_col] = df[merge_cols].bfill(axis=1).iloc[:, 0]

    # Delete all perplexity_cols
    df.drop(columns=[*merge_cols, *delete_cols], inplace=True)

    return df


# Merge all columns with col_7b_pattern
olmo_7b = clean_df(olmo_7b, merge_patterns_7b, delete_patterns_7b, 'perplexity')
olmo_1b = clean_df(olmo_1b, merge_patterns_1b, delete_patterns_1b, 'perplexity')

# Plot 