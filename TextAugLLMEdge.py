#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TextAugLLMEdge

# Supplementary Results

# Partha Pratim Ray, 15/6/2025

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your CSV file (change if needed)
csv_path = r"D:\technical writing\textaugmentation\result.csv"

# Load data
df = pd.read_csv(csv_path)
print("Columns:", df.columns.tolist())
print(df.head())


# In[2]:


# BAR PLOT
df['total_duration_s'] = df['total_duration_ns'] / 1e9
agg = df.groupby(['augmentation_type', 'model'])['total_duration_s'].mean().reset_index()
augmentation_order = sorted(df['augmentation_type'].unique())

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='total_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Total Duration (s)", fontsize=12)
plt.title("Mean Inference Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(2.5,20.5)
plt.show()

print("Mean Total Duration (s) Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
violin = sns.violinplot(
    data=df,
    x='augmentation_type',
    y='total_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"  # Remove split=True!
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Total Duration (s)", fontsize=12)
plt.title("Distribution of Inference Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[3]:


# BAR PLOT
df['load_duration_s'] = df['load_duration_ns'] / 1e9
agg = df.groupby(['augmentation_type', 'model'])['load_duration_s'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='load_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Load Duration (s)", fontsize=12)
plt.title("Mean Load Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Load Duration (s) Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
violin = sns.violinplot(
    data=df,
    x='augmentation_type',
    y='load_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Load Duration (s)", fontsize=12)
plt.title("Distribution of Load Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[4]:


# BAR PLOT
agg = df.groupby(['augmentation_type', 'model'])['prompt_eval_count'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='prompt_eval_count',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.1f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Prompt Eval Count", fontsize=12)
plt.title("Mean Prompt Eval Count by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(40,125)
plt.show()

print("Mean Prompt Eval Count Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
violin = sns.violinplot(
    data=df,
    x='augmentation_type',
    y='prompt_eval_count',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Prompt Eval Count", fontsize=12)
plt.title("Distribution of Prompt Eval Count by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[5]:


# BAR PLOT
df['prompt_eval_duration_s'] = df['prompt_eval_duration_ns'] / 1e9
agg = df.groupby(['augmentation_type', 'model'])['prompt_eval_duration_s'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='prompt_eval_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Prompt Eval Duration (s)", fontsize=12)
plt.title("Mean Prompt Eval Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(1,3.5)
plt.show()

print("Mean Prompt Eval Duration (s) Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
violin = sns.violinplot(
    data=df,
    x='augmentation_type',
    y='prompt_eval_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Prompt Eval Duration (s)", fontsize=12)
plt.title("Distribution of Prompt Eval Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[6]:


# BAR PLOT
agg = df.groupby(['augmentation_type', 'model'])['eval_count'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='eval_count',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.1f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Eval Count", fontsize=12)
plt.title("Mean Eval Count by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(20,65)
plt.show()

print("Mean Eval Count Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
violin = sns.violinplot(
    data=df,
    x='augmentation_type',
    y='eval_count',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Eval Count", fontsize=12)
plt.title("Distribution of Eval Count by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[7]:


# BAR PLOT
df['eval_duration_s'] = df['eval_duration_ns'] / 1e9
agg = df.groupby(['augmentation_type', 'model'])['eval_duration_s'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='eval_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Eval Duration (s)", fontsize=12)
plt.title("Mean Eval Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(2,15)
plt.show()

print("Mean Eval Duration (s) Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
violin = sns.violinplot(
    data=df,
    x='augmentation_type',
    y='eval_duration_s',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Eval Duration (s)", fontsize=12)
plt.title("Distribution of Eval Duration by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[8]:


# BAR PLOT
agg = df.groupby(['augmentation_type', 'model'])['tokens_per_second'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='tokens_per_second',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Tokens per Second", fontsize=12)
plt.title("Tokens per Second by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(4,10)
plt.show()

print("Tokens per Second Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
violin = sns.violinplot(
    data=df,
    x='augmentation_type',
    y='tokens_per_second',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Tokens per Second", fontsize=12)
plt.title("Distribution of Tokens per Second by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[9]:


agg = df.groupby(['augmentation_type', 'model'])['levenshtein_similarity'].mean().reset_index()
augmentation_order = sorted(df['augmentation_type'].unique())

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='levenshtein_similarity',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Levenshtein Similarity", fontsize=12)
plt.title("Levenshtein Similarity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Levenshtein Similarity Table:\n")
print(agg.to_string(index=False))

# VIOLIN PLOT
plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='levenshtein_similarity',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Levenshtein Similarity", fontsize=12)
plt.title("Distribution of Levenshtein Similarity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()



# In[10]:


agg = df.groupby(['augmentation_type', 'model'])['jaccard_similarity'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='jaccard_similarity',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Jaccard Similarity", fontsize=12)
plt.title("Jaccard Similarity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Jaccard Similarity Table:\n")
print(agg.to_string(index=False))


plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='jaccard_similarity',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Jaccard Similarity", fontsize=12)
plt.title("Distribution of Jaccard Similarity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[11]:


agg = df.groupby(['augmentation_type', 'model'])['length_ratio'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='length_ratio',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Length Ratio", fontsize=12)
plt.title("Length Ratio by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Length Ratio Table:\n")
print(agg.to_string(index=False))

plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='length_ratio',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Length Ratio", fontsize=12)
plt.title("Distribution of Length Ratio by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(.4, 1.6)
plt.show()


# In[12]:


agg = df.groupby(['augmentation_type', 'model'])['bleu'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='bleu',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean BLEU Score", fontsize=12)
plt.title("BLEU Score by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean BLEU Score Table:\n")
print(agg.to_string(index=False))

plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='bleu',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("BLEU Score", fontsize=12)
plt.title("Distribution of BLEU Score by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[13]:


agg = df.groupby(['augmentation_type', 'model'])['cosine_similarity'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='cosine_similarity',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Cosine Similarity", fontsize=12)
plt.title("Cosine Similarity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Cosine Similarity Table:\n")
print(agg.to_string(index=False))

plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='cosine_similarity',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Cosine Similarity", fontsize=12)
plt.title("Distribution of Cosine Similarity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[14]:


agg = df.groupby(['augmentation_type', 'model'])['wer'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='wer',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Word Error Rate", fontsize=12)
plt.title("Word Error Rate by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Word Error Rate Table:\n")
print(agg.to_string(index=False))

plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='wer',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Word Error Rate", fontsize=12)
plt.title("Distribution of Word Error Rate by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[15]:


agg = df.groupby(['augmentation_type', 'model'])['char_diversity'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='char_diversity',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Character Diversity", fontsize=12)
plt.title("Character Diversity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Character Diversity Table:\n")
print(agg.to_string(index=False))

plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='char_diversity',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Character Diversity", fontsize=12)
plt.title("Distribution of Character Diversity by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[16]:


agg = df.groupby(['augmentation_type', 'model'])['type_token_ratio'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='type_token_ratio',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Type Token Ratio", fontsize=12)
plt.title("Type Token Ratio by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.ylim(.8,1.01)
plt.show()

print("Mean Type Token Ratio Table:\n")
print(agg.to_string(index=False))

plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='type_token_ratio',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Type Token Ratio", fontsize=12)
plt.title("Distribution of Type Token Ratio by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[17]:


agg = df.groupby(['augmentation_type', 'model'])['bigram_overlap'].mean().reset_index()

plt.figure(figsize=(12,7))
bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='bigram_overlap',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.3f', label_type='edge', padding=2, fontsize=6)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Bigram Overlap", fontsize=12)
plt.title("Bigram Overlap by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Bigram Overlap Table:\n")
print(agg.to_string(index=False))

plt.figure(figsize=(12,7))
sns.violinplot(
    data=df,
    x='augmentation_type',
    y='bigram_overlap',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)
plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Bigram Overlap", fontsize=12)
plt.title("Distribution of Bigram Overlap by Augmentation Type and LLM", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV if the DataFrame is not currently available
if 'df' not in globals():
    csv_path = r"D:\technical writing\textaugmentation\result.csv"
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully: {df.shape}")

# Ensure duration columns exist (nanoseconds to seconds)
duration_columns = {
    'total_duration_s': 'total_duration_ns',
    'load_duration_s': 'load_duration_ns',
    'prompt_eval_duration_s': 'prompt_eval_duration_ns'
}

for seconds_col, nanos_col in duration_columns.items():
    if seconds_col not in df.columns and nanos_col in df.columns:
        df[seconds_col] = pd.to_numeric(
            df[nanos_col], errors='coerce'
        ) / 1e9

# Define correlation metrics
metrics = [
    'levenshtein_similarity',
    'jaccard_similarity',
    'length_ratio',
    'bleu',
    'cosine_similarity',
    'wer',
    'char_diversity',
    'type_token_ratio',
    'bigram_overlap',
    'total_duration_s',
    'load_duration_s',
    'prompt_eval_duration_s',
    'tokens_per_second'
]

# Check availability
available_metrics = [m for m in metrics if m in df.columns]
missing_metrics = [m for m in metrics if m not in df.columns]

if missing_metrics:
    print("Missing metrics:", missing_metrics)

if len(available_metrics) < 2:
    raise ValueError("At least two metric columns are required.")

# Convert available metrics to numeric values
metric_data = df[available_metrics].apply(
    pd.to_numeric, errors='coerce'
)

# Pearson correlation matrix
corr = metric_data.corr(method='pearson')

# Plot heatmap
plt.figure(figsize=(13, 11))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)

plt.title(
    "Correlation Heatmap of Text Augmentation and Performance Metrics",
    fontsize=14,
    pad=20
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nCorrelation Matrix:")
display(corr.round(3))


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define augmentation order
augmentation_order = sorted(
    df['augmentation_type'].dropna().unique()
)

# 2. Normalize tokens per second
tps_min = df['tokens_per_second'].min()
tps_max = df['tokens_per_second'].max()

if tps_max > tps_min:
    df['tokens_per_second_norm'] = (
        (df['tokens_per_second'] - tps_min) /
        (tps_max - tps_min)
    )
else:
    df['tokens_per_second_norm'] = 0.0

# 3. Compute composite score
w1, w2, w3 = 0.4, 0.3, 0.3

df['composite_score'] = (
    w1 * df['levenshtein_similarity'] +
    w2 * (1 - df['bigram_overlap']) +
    w3 * df['tokens_per_second_norm']
)

# 4. Aggregate by augmentation type and model
agg = (
    df.groupby(['augmentation_type', 'model'])['composite_score']
    .mean()
    .reset_index()
)

# 5. Bar plot
plt.figure(figsize=(12, 7))

bar = sns.barplot(
    data=agg,
    x='augmentation_type',
    y='composite_score',
    hue='model',
    order=augmentation_order,
    palette='tab10'
)

for container in bar.containers:
    bar.bar_label(
        container,
        fmt='%.2f',
        label_type='edge',
        padding=2,
        fontsize=7
    )

plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Mean Composite Score", fontsize=12)
plt.title(
    "Composite Score (Fidelity, Novelty, Speed) by Augmentation and LLM",
    fontsize=14
)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Composite Score Table:\n")
print(agg.to_string(index=False))

# 6. Violin plot
plt.figure(figsize=(12, 7))

sns.violinplot(
    data=df,
    x='augmentation_type',
    y='composite_score',
    hue='model',
    order=augmentation_order,
    palette='tab10',
    inner="quartile"
)

plt.xlabel("Augmentation Type", fontsize=12)
plt.ylabel("Composite Score", fontsize=12)
plt.title(
    "Composite Score Distribution by Augmentation and LLM",
    fontsize=14
)
plt.xticks(rotation=45)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()


# In[20]:


metric_col = 'bleu'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean BLEU", fontsize=12)
plt.title("Mean BLEU by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean BLEU Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[21]:


metric_col = 'levenshtein_similarity'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Levenshtein Similarity", fontsize=12)
plt.title("Mean Levenshtein Similarity by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Levenshtein Similarity Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[22]:


metric_col = 'jaccard_similarity'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Jaccard Similarity", fontsize=12)
plt.title("Mean Jaccard Similarity by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Jaccard Similarity Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[23]:


metric_col = 'length_ratio'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Length Ratio", fontsize=12)
plt.title("Mean Length Ratio by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Length Ratio Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[24]:


metric_col = 'cosine_similarity'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Cosine Similarity", fontsize=12)
plt.title("Mean Cosine Similarity by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Cosine Similarity Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[25]:


metric_col = 'wer'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Word Error Rate", fontsize=12)
plt.title("Mean Word Error Rate by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Word Error Rate Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[26]:


metric_col = 'char_diversity'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Character Diversity", fontsize=12)
plt.title("Mean Character Diversity by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Character Diversity Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[27]:


metric_col = 'type_token_ratio'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Type Token Ratio", fontsize=12)
plt.title("Mean Type Token Ratio by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Type Token Ratio Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[28]:


metric_col = 'bigram_overlap'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.3f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Bigram Overlap", fontsize=12)
plt.title("Mean Bigram Overlap by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Bigram Overlap Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[29]:


metric_col = 'total_duration_s'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Total Duration (s)", fontsize=12)
plt.title("Mean Total Duration (s) by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Total Duration (s) Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[30]:


df['load_duration_s'] = df['load_duration_ns'] / 1e9
metric_col = 'load_duration_s'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Load Duration (s)", fontsize=12)
plt.title("Mean Load Duration (s) by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Load Duration (s) Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[31]:


metric_col = 'prompt_eval_count'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.1f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Prompt Eval Count", fontsize=12)
plt.title("Mean Prompt Eval Count by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Prompt Eval Count Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[32]:


df['prompt_eval_duration_s'] = df['prompt_eval_duration_ns'] / 1e9
metric_col = 'prompt_eval_duration_s'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Prompt Eval Duration (s)", fontsize=12)
plt.title("Mean Prompt Eval Duration (s) by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Prompt Eval Duration (s) Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[33]:


metric_col = 'eval_count'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.1f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Eval Count", fontsize=12)
plt.title("Mean Eval Count by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Eval Count Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[34]:


df['eval_duration_s'] = df['eval_duration_ns'] / 1e9
metric_col = 'eval_duration_s'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Eval Duration (s)", fontsize=12)
plt.title("Mean Eval Duration (s) by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Eval Duration (s) Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[35]:


metric_col = 'tokens_per_second'
agg = df.groupby(['prompt_id', 'model'])[metric_col].mean().reset_index()
prompt_order = sorted(df['prompt_id'].unique())

plt.figure(figsize=(10,6))
bar = sns.barplot(
    data=agg,
    x='prompt_id',
    y=metric_col,
    hue='model',
    order=prompt_order,
    palette='tab10'
)
for container in bar.containers:
    bar.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=7)
plt.xlabel("Prompt ID", fontsize=12)
plt.ylabel("Mean Tokens per Second", fontsize=12)
plt.title("Mean Tokens per Second by Prompt and Model", fontsize=14)
plt.legend(title="LLM")
plt.tight_layout()
plt.show()

print("Mean Tokens per Second Table by Prompt and Model:\n")
print(agg.to_string(index=False))


# In[36]:


from scipy.stats import f_oneway
import pandas as pd

metric_col = 'bleu'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (BLEU):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean BLEU across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean BLEU across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[37]:


metric_col = 'levenshtein_similarity'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Levenshtein Similarity):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Levenshtein similarity across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Levenshtein similarity across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[38]:


metric_col = 'jaccard_similarity'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Jaccard Similarity):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Jaccard similarity across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Jaccard similarity across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[39]:


metric_col = 'length_ratio'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Length Ratio):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Length Ratio across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Length Ratio across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[40]:


metric_col = 'cosine_similarity'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Cosine Similarity):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Cosine Similarity across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Cosine Similarity across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[41]:


metric_col = 'wer'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Word Error Rate):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean WER across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean WER across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[42]:


metric_col = 'char_diversity'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Character Diversity):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Character Diversity across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Character Diversity across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[43]:


metric_col = 'type_token_ratio'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Type Token Ratio):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Type Token Ratio across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Type Token Ratio across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[44]:


metric_col = 'bigram_overlap'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Bigram Overlap):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Bigram Overlap across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Bigram Overlap across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[45]:


metric_col = 'total_duration_s'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Total Duration (s)):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean total duration (s) across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean total duration (s) across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[46]:


metric_col = 'prompt_eval_count'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Prompt Eval Count):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Prompt Eval Count across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Prompt Eval Count across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[47]:


metric_col = 'prompt_eval_duration_ns'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Prompt Eval Duration [ns]):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Prompt Eval Duration (ns) across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Prompt Eval Duration (ns) across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[48]:


metric_col = 'eval_count'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Eval Count):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Eval Count across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Eval Count across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[49]:


metric_col = 'eval_duration_ns'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Eval Duration [ns]):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Eval Duration (ns) across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Eval Duration (ns) across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[50]:


metric_col = 'tokens_per_second'
anova_results = []
for aug in sorted(df['augmentation_type'].unique()):
    subset = df[df['augmentation_type'] == aug]
    groups = [subset[subset['model'] == m][metric_col].dropna().values
              for m in subset['model'].unique()]
    if sum([len(g) > 1 for g in groups]) >= 2:
        stat, pval = f_oneway(*groups)
        anova_results.append({'augmentation_type': aug, 'F-statistic': stat, 'p-value': pval})
    else:
        anova_results.append({'augmentation_type': aug, 'F-statistic': float('nan'), 'p-value': float('nan')})

anova_df = pd.DataFrame(anova_results)
anova_df['Significant'] = anova_df['p-value'] < 0.05
display(anova_df)

print("\nSignificance Interpretation (Tokens per Second):")
for idx, row in anova_df.iterrows():
    if pd.notna(row['p-value']):
        if row['Significant']:
            print(f"Augmentation '{row['augmentation_type']}': Significant difference in mean Tokens per Second across LLMs (p={row['p-value']:.4g}).")
        else:
            print(f"Augmentation '{row['augmentation_type']}': No significant difference in mean Tokens per Second across LLMs (p={row['p-value']:.4g}).")
    else:
        print(f"Augmentation '{row['augmentation_type']}': Insufficient data for ANOVA.")


# In[51]:


# This code runs the Kruskal–Wallis test for each metric, grouped by augmentation_type, and does pairwise Mann–Whitney U tests if significant.
# It prints results for each metric and group.

# Interpretation:
# If p < 0.05 in Kruskal–Wallis, at least one model is different for that metric/augmentation.
# Mann–Whitney U will tell you which LLM pairs differ significantly.

from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations
import numpy as np

metrics = [
    'levenshtein_similarity', 'jaccard_similarity', 'length_ratio', 'bleu',
    'cosine_similarity', 'wer', 'char_diversity', 'type_token_ratio', 'bigram_overlap',
    'total_duration_ns', 'load_duration_ns', 'prompt_eval_count', 'prompt_eval_duration_ns',
    'eval_count', 'eval_duration_ns', 'tokens_per_second'
]

group_var = 'augmentation_type'  # Or 'prompt_id'

for metric_col in metrics:
    print(f"\n\n==== Kruskal–Wallis and Mann–Whitney U for {metric_col} (by {group_var}) ====")
    for group_val in sorted(df[group_var].unique()):
        sub = df[df[group_var] == group_val]
        models = sub['model'].unique()
        data = [sub[sub['model'] == m][metric_col].dropna().values for m in models]
        if len(models) > 1:
            # Flatten all values for all groups, ignore nan
            all_values = np.concatenate([d for d in data if len(d) > 0])
            if len(all_values) == 0:
                print(f"  {group_var.capitalize()} '{group_val}': No data available.")
                continue
            # Check if all values are identical
            if np.all(all_values == all_values[0]):
                print(f"  {group_var.capitalize()} '{group_val}': All values identical; cannot perform Kruskal–Wallis test.")
                continue
            stat, p = kruskal(*data)
            print(f"{group_var.capitalize()} '{group_val}': Kruskal–Wallis H={stat:.3f}, p={p:.4g}")
            if p < 0.05:
                print("  Significant: Pairwise comparisons:")
                for m1, m2 in combinations(models, 2):
                    d1 = sub[sub['model'] == m1][metric_col].dropna()
                    d2 = sub[sub['model'] == m2][metric_col].dropna()
                    # Mann–Whitney also needs at least one unique value
                    if len(d1) > 0 and len(d2) > 0 and (not np.all(d1 == d1.iloc[0]) or not np.all(d2 == d2.iloc[0])):
                        try:
                            u, p_mw = mannwhitneyu(d1, d2, alternative='two-sided')
                            print(f"    {m1} vs {m2}: U={u:.1f}, p={p_mw:.4g}")
                        except ValueError as e:
                            print(f"    {m1} vs {m2}: Mann–Whitney not possible ({e})")
            else:
                print("  No significant difference among LLMs.")
        else:
            print(f"{group_var.capitalize()} '{group_val}': Only one model present.")


# In[52]:


# Friedman’s Test (by augmentation_type)

from scipy.stats import friedmanchisquare

metrics = [
    'levenshtein_similarity', 'jaccard_similarity', 'length_ratio', 'bleu',
    'cosine_similarity', 'wer', 'char_diversity', 'type_token_ratio', 'bigram_overlap',
    'total_duration_ns', 'load_duration_ns', 'prompt_eval_count', 'prompt_eval_duration_ns',
    'eval_count', 'eval_duration_ns', 'tokens_per_second'
]

group_var = 'augmentation_type'   # Or use 'prompt_id' if you want

for metric_col in metrics:
    print(f"\n=== Friedman’s Test for {metric_col} (by {group_var}) ===")
    for group_val in sorted(df[group_var].unique()):
        sub = df[df[group_var] == group_val]
        pivot = sub.pivot(index='prompt_id', columns='model', values=metric_col)
        # Only run if all models have data for all prompt_id
        if pivot.notnull().all(axis=1).any() and pivot.shape[1] > 1:
            # Only keep rows (prompts) where all models present
            data = [pivot[m].dropna().values for m in pivot.columns]
            if all(len(x) == len(data[0]) for x in data):
                stat, p = friedmanchisquare(*data)
                print(f"{group_var.capitalize()} '{group_val}': Friedman's chi2={stat:.3f}, p={p:.4g}")
            else:
                print(f"{group_var.capitalize()} '{group_val}': Not all models have data for all prompts.")
        else:
            print(f"{group_var.capitalize()} '{group_val}': Not enough data for Friedman’s test.")
# 


# In[53]:


# Kolmogorov–Smirnov (K–S) Test (for all pairs of models)

from scipy.stats import ks_2samp
from itertools import combinations

for metric_col in metrics:
    print(f"\n=== Kolmogorov–Smirnov Test for {metric_col} (All Model Pairs) ===")
    models = df['model'].unique()
    for m1, m2 in combinations(models, 2):
        data1 = df[df['model'] == m1][metric_col].dropna()
        data2 = df[df['model'] == m2][metric_col].dropna()
        if len(data1) > 0 and len(data2) > 0:
            ks_stat, ks_p = ks_2samp(data1, data2)
            print(f"  {m1} vs {m2}: KS stat={ks_stat:.3f}, p={ks_p:.4g}")
        else:
            print(f"  {m1} vs {m2}: Not enough data.")


# In[54]:


# Spearman’s Rank Correlation for All Metrics

# How to interpret:

# Correlation coefficient in [-1, 1]:
# Closer to 1 = strong positive monotonic relation
# Closer to -1 = strong negative
# Close to 0 = no monotonic association

# P-value:
# p < 0.05 means significant correlation
# p > 0.05 means correlation is not statistically significant

import pandas as pd
from scipy.stats import spearmanr

# List of your numeric metrics
metrics = [
    'levenshtein_similarity', 'jaccard_similarity', 'length_ratio', 'bleu',
    'cosine_similarity', 'wer', 'char_diversity', 'type_token_ratio', 'bigram_overlap',
    'total_duration_ns', 'load_duration_ns', 'prompt_eval_count', 'prompt_eval_duration_ns',
    'eval_count', 'eval_duration_ns', 'tokens_per_second'
]

# Subset dataframe to just these columns, drop rows with any NaN
metrics_df = df[metrics].dropna()

# Compute Spearman correlation (returns correlation matrix and p-value matrix)
corr, pval = spearmanr(metrics_df, axis=0)

# To DataFrame for easier reading:
corr_df = pd.DataFrame(corr, index=metrics, columns=metrics)
pval_df = pd.DataFrame(pval, index=metrics, columns=metrics)

# Print or display the correlation matrix
print("Spearman Correlation Matrix:")
print(corr_df.round(3))

print("\nSpearman Correlation P-value Matrix:")
print(pval_df.round(3))


# In[55]:


import pandas as pd

metrics = [
    'levenshtein_similarity', 'jaccard_similarity', 'length_ratio', 'bleu',
    'cosine_similarity', 'wer', 'char_diversity', 'type_token_ratio', 'bigram_overlap',
    'total_duration_ns', 'load_duration_ns', 'prompt_eval_count', 'prompt_eval_duration_ns',
    'eval_count', 'eval_duration_ns', 'tokens_per_second'
]

# Filter to relevant columns, drop rows with any missing value in those columns
df_manova = df[['model', 'augmentation_type'] + metrics].dropna()
print(df_manova.shape)


## MANOVA by Model Only

# Interpretation
# Look at the line for model in Wilks’ lambda, Pillai’s trace, Hotelling-Lawley trace, and Roy’s largest root.
# The Pr > F column gives the p-value for the global null: "All LLMs have the same mean vector of metrics."
# p < 0.05 means at least one LLM is different on at least one combination of metrics.

from statsmodels.multivariate.manova import MANOVA

formula = ' + '.join(metrics) + ' ~ model'
maov = MANOVA.from_formula(formula, data=df_manova)
print(maov.mv_test())


# In[56]:


#MANOVA by Model and Augmentation Type (Factorial MANOVA)

formula = ' + '.join(metrics) + ' ~ model + augmentation_type'
maov2 = MANOVA.from_formula(formula, data=df_manova)
print(maov2.mv_test())


# In[57]:


# MANOVA With Interaction Term

formula = ' + '.join(metrics) + ' ~ model * augmentation_type'
maov3 = MANOVA.from_formula(formula, data=df_manova)
print(maov3.mv_test())


# In[58]:


#Post-hoc Exploration: Which Metrics Are Responsible?

#If MANOVA is significant, run univariate ANOVA for each metric:


import statsmodels.api as sm
from statsmodels.formula.api import ols

for metric in metrics:
    model = ols(f'{metric} ~ model', data=df_manova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"ANOVA for {metric}:\n", anova_table, "\n")


# In[59]:


# Interpreting Model-wise Effect Sizes
#1. Mean Difference (mean_diff):
#This is simply the difference in mean value of a metric between two models.

#Positive value: model_1 has a higher mean than model_2 for that metric.

#Negative value: model_2 has a higher mean.

#2. Cohen’s d (cohens_d):
#Cohen’s d quantifies the magnitude of the difference (in standard deviations) between two distributions.

#Conventional thresholds for interpretation:

#    |d| < 0.2: Negligible difference (practically equivalent)

#    0.2 ≤ |d| < 0.5: Small effect (slight difference)

#     0.5 ≤ |d| < 0.8: Medium effect (moderate difference)

#   |d| ≥ 0.8: Large effect (substantial/practically meaningful difference)

# Sign of d:

# Positive: model_1 performs better (higher mean for positively oriented metrics, e.g., similarity, tokens/sec).

# Negative: model_2 performs better.

# 3. Metric Orientation
# For similarity/accuracy metrics (bleu, levenshtein_similarity, cosine_similarity, etc.): Higher is better.

# For error or duration metrics (wer, total_duration_ns, etc.): Lower is better.

# If Cohen’s d is positive and the metric is an error/duration, model_1 has worse performance.



# In[60]:


import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

metrics = [
    "total_duration_ns", "load_duration_ns", "prompt_eval_count", "prompt_eval_duration_ns", "eval_count",
    "eval_duration_ns", "tokens_per_second", "levenshtein_similarity", "jaccard_similarity", "length_ratio",
    "bleu", "cosine_similarity", "wer", "char_diversity", "type_token_ratio", "bigram_overlap"
]

model_list = sorted(df['model'].unique())
results = []


for metric in metrics:
    for m1, m2 in combinations(model_list, 2):
        x = df[df['model'] == m1][metric].dropna()
        y = df[df['model'] == m2][metric].dropna()
        if len(x) > 1 and len(y) > 1:
            d = cohens_d(x, y)
            mean_diff = x.mean() - y.mean()
            t_stat, p_val = ttest_ind(x, y, equal_var=False)
            results.append({
                'metric': metric,
                'model_1': m1,
                'model_2': m2,
                'mean_model_1': x.mean(),
                'mean_model_2': y.mean(),
                'std_model_1': x.std(ddof=1),
                'std_model_2': y.std(ddof=1),
                'n_model_1': len(x),
                'n_model_2': len(y),
                'mean_diff': mean_diff,
                'cohens_d': d,
                'ttest_pval': p_val
            })

effectsize_df = pd.DataFrame(results)

pd.set_option('display.max_rows', None)
print("Table: Pairwise Model Comparisons Across All Metrics—Means, Standard Deviations, Sample Sizes, Mean Differences, Cohen’s d, and t-Test p-values")
display(effectsize_df)


# In[61]:


#What does Cliff’s Delta calculate?
#It measures the probability that a randomly selected value from group A will be larger than a randomly selected value from group B, minus the reverse probability.

#Values range from –1 (all values in A are less than all in B) to +1 (all values in A are greater than all in B).

#0 means the two groups completely overlap (no effect).

#Positive values: A tends to be greater than B.

#Negative values: A tends to be less than B.


# Interpretation
# Large effect (|delta| ≥ 0.474): Models are substantially different for this metric

# Medium/small effect: Moderate/small practical differences

# Negligible: Distributions largely overlap



# In[62]:


import pandas as pd
import numpy as np
from itertools import combinations

# --- Pure Python Cliff's Delta implementation ---
def cliffs_delta(x, y):
    """
    Computes Cliff's delta and magnitude for two arrays/lists x, y.
    Returns delta (float) and magnitude (str: 'negligible', 'small', 'medium', 'large')
    """
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    more = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    delta = (more - less) / (n_x * n_y)
    # Magnitude thresholds (Romano et al., 2006)
    adelta = abs(delta)
    if adelta < 0.147:
        magnitude = "negligible"
    elif adelta < 0.33:
        magnitude = "small"
    elif adelta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    return delta, magnitude



# --- Specify metrics for which to compute Cliff's delta ---
metrics = [
    "total_duration_ns", "load_duration_ns", "prompt_eval_count", "prompt_eval_duration_ns", "eval_count",
    "eval_duration_ns", "tokens_per_second", "levenshtein_similarity", "jaccard_similarity", "length_ratio",
    "bleu", "cosine_similarity", "wer", "char_diversity", "type_token_ratio", "bigram_overlap"
]

# --- Get unique models ---
model_list = sorted(df['model'].unique())

# --- Compute Cliff's delta for all pairs and all metrics ---
results = []
for metric in metrics:
    for m1, m2 in combinations(model_list, 2):
        x = df[df['model'] == m1][metric].dropna()
        y = df[df['model'] == m2][metric].dropna()
        if len(x) > 1 and len(y) > 1:
            delta, magnitude = cliffs_delta(x, y)
            results.append({
                'metric': metric,
                'model_1': m1,
                'model_2': m2,
                "cliffs_delta": delta,
                "magnitude": magnitude,
                "mean_model_1": x.mean(),
                "mean_model_2": y.mean(),
                "n_model_1": len(x),
                "n_model_2": len(y)
            })

cliffs_df = pd.DataFrame(results)

# Show all rows (optional, for Jupyter)
pd.set_option("display.max_rows", None)
print("Table: Cliff’s Delta Effect Sizes for Pairwise Model Comparisons Across All Metrics")
display(cliffs_df)


# In[63]:


#Association between Text Augmentation with Prompts

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

# Optional: If you don’t have `theils_u` function, define it:
def theils_u(x, y):
    """Theil's U (uncertainty coefficient U(x|y)), asymmetric"""
    s_xy = mutual_info_score(x, y)
    s_x = entropy_from_series(x)
    return 0 if s_x == 0 else s_xy / s_x

def entropy_from_series(s):
    p = s.value_counts(normalize=True)
    return -np.sum(p * np.log2(p + 1e-12))

# Build contingency table
contingency = pd.crosstab(df['augmentation_type'], df['prompt_id'])

# 1. Chi-square Test
chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency)

# 2. Cramér's V
n = contingency.values.sum()
cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))

# 3. Mutual Information Score
mi = mutual_info_score(df['augmentation_type'], df['prompt_id'])

# 4. Theil's U (both directions, asymmetric)
u_aug_given_prompt = theils_u(df['augmentation_type'], df['prompt_id'])
u_prompt_given_aug = theils_u(df['prompt_id'], df['augmentation_type'])

# 5. Summary Table 
summary = pd.DataFrame({
    'Test': [
        "Chi-square Test of Independence",
        "Cramér's V (Effect Size)",
        "Mutual Information Score",
        "Theil's U (Augmentation|Prompt)",
        "Theil's U (Prompt|Augmentation)"
    ],
    'Statistic/Value': [
        f"Chi2={chi2_stat:.5g}, dof={dof}, p={p_chi2:.5g}",
        f"{cramers_v:.5f}",
        f"{mi:.5f}",
        f"{u_aug_given_prompt:.5f}",
        f"{u_prompt_given_aug:.5f}"
    ]
})

print("Association between Text Augmentation with Prompts")
display(summary)


# In[64]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Select Numeric Columns for PCA
numeric_cols = [
    'total_duration_ns', 'load_duration_ns', 'prompt_eval_count',
    'prompt_eval_duration_ns', 'eval_count', 'eval_duration_ns',
    'tokens_per_second', 'levenshtein_similarity', 'jaccard_similarity',
    'length_ratio', 'bleu', 'cosine_similarity', 'wer', 'char_diversity',
    'type_token_ratio', 'bigram_overlap'
]
df_numeric = df[numeric_cols].dropna()

# 3. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# 4. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# 5. KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
df_pca['Cluster'] = clusters

# 6. Plotting
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=80)
plt.title('KMeans Clustering on PCA-Reduced Augmentation Metrics')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()



# Print the cluster assignments for the first few rows and the centroids
print("Cluster assignments (first 10):")
print(df_pca['Cluster'].head(10))

print("\nKMeans cluster centers in PCA space:")
print(kmeans.cluster_centers_)

# Count the number of points in each cluster
print("\nNumber of points in each cluster:")
print(df_pca['Cluster'].value_counts())

loadings = pd.DataFrame(
    pca.components_.T,  # shape: (n_features, n_components)
    index=numeric_cols,
    columns=['PC1', 'PC2']
)
print(loadings)


# In[ ]:




