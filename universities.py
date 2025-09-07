#%%
import src.google_search as google_search
import pandas as pd
from tqdm import tqdm
import seaborn as sns

#%%
universities_df = pd.read_csv('data/download/partners.csv')

#%%
import matplotlib.pyplot as plt

# ensure numeric columns and drop invalid rows
plot_df = universities_df.copy()
plot_df['students_amount'] = pd.to_numeric(plot_df['students_amount'], errors='coerce')
plot_df['teachers_amount'] = pd.to_numeric(plot_df['teachers_amount'], errors='coerce')
plot_df = plot_df.dropna(subset=['students_amount', 'teachers_amount'])

# scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=plot_df, x='students_amount', y='teachers_amount', s=20)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Количество студентов')
plt.ylabel('Количество преподавателей')
plt.title('Соотношение студентов и преподавателей в университетах')
plt.grid(alpha=0.3)

# annotate a few largest universities by students
labeled = set()
for _, r in plot_df.nlargest(9, 'students_amount').iterrows():
    label = r.get('abbreviation') or r.get('name') or ''
    plt.text(r['students_amount'], r['teachers_amount'], label, fontsize=6)
    labeled.add(r.name)

# compute teacher-to-student ratio and annotate lowest/highest 5
plot_df = plot_df[(plot_df['students_amount'] > 100) & (plot_df['teachers_amount'] > 5)].copy()
plot_df['ratio'] = plot_df['teachers_amount'] / plot_df['students_amount']

# annotate 5 universities with lowest ratio (few teachers per student)
for _, r in plot_df.nsmallest(5, 'ratio').iterrows():
    if r.name in labeled:
        continue
    label = r.get('abbreviation') or r.get('name') or ''
    plt.text(r['students_amount'] * 1.05, r['teachers_amount'] * 1.05, label, fontsize=6, color='red')
    labeled.add(r.name)

# annotate 5 universities with highest ratio (many teachers per student)
for _, r in plot_df.nlargest(5, 'ratio').iterrows():
    if r.name in labeled:
        continue
    label = r.get('abbreviation') or r.get('name') or ''
    plt.text(r['students_amount'] * 1.05, r['teachers_amount'] * 1.05, label, fontsize=6, color='red')
    labeled.add(r.name)

plt.tight_layout()
plt.show()

#%%
# find university websites
urls = []
for (i, row) in tqdm(universities_df.iterrows(), total=len(universities_df)):
    university_name = row['name']
    query = f"{university_name}"
    try:
        results = google_search.search(query, rate_limit=0)
        url = results[0]['url']
    except Exception as e:
        print(f"Error for {university_name}: {e}")
        url = ""
    urls.append(url)
    print(row['abbreviation'], url)

universities_df['url'] = urls
universities_df.to_csv('data/generated/universities.csv', index=False)