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
    university_abbreviation = row['abbreviation'] if pd.notna(row['abbreviation']) else ""
    university_name = row['name']
    university_town = row['town'] if pd.notna(row['town']) else ""
    query = f"{university_abbreviation} {university_name} {university_town}"
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

#%%
universities_df = pd.read_csv("data/generated/universities.csv")
print(len(universities_df))
# ensure students_amount and teachers_amount are ints, convert NaNs to 0, drop rows where either is 0
for col in ['students_amount', 'teachers_amount']:
    universities_df[col] = pd.to_numeric(universities_df[col], errors='coerce').fillna(0).astype(int)

universities_df = universities_df[(universities_df['students_amount'] >= 100) & (universities_df['teachers_amount'] >= 10)].copy().reset_index(drop=True)
print(len(universities_df))

#%%
# merge rows with exactly the same name: sum students/teachers, keep abbreviation and url from row with most students
merged = []
for name, group in universities_df.groupby('name', sort=False):
    if pd.isna(name) or name == "":
        # keep as-is (no meaningful name to merge on)
        merged.extend(group.to_dict('records'))
        continue

    if len(group) == 1:
        merged.append(group.iloc[0].to_dict())
        continue

    total_students = int(group['students_amount'].sum())
    total_teachers = int(group['teachers_amount'].sum())

    # choose row with most students to keep abbreviation and url
    top_idx = group['students_amount'].idxmax()
    top_row = group.loc[top_idx].copy()
    top_row['students_amount'] = total_students
    top_row['teachers_amount'] = total_teachers

    merged.append(top_row.to_dict())

new_df = pd.DataFrame(merged).reset_index(drop=True)

print(f"Merged {len(universities_df) - len(new_df)} rows -> {len(new_df)} rows remaining.")
universities_df = new_df
print(len(universities_df))

#%%
from urllib.parse import urlsplit
import tldextract

HOST_DROP_ETLD1 = {
    "wikipedia.org", "vk.com", "facebook.com", "instagram.com", "ok.ru", "yandex.ru",
    "youtube.com", "t.me", "vuzopedia.ru", "postupi.online", "universitys.ru", "ucheba.ru", "ivobr.ru", "ciur.ru"
}
COLLAPSE_PREFIXES = {"www", "en", "ru", "pk", "abitur", "priem", "admission", "admit", "new"}
SSUZ_HINTS = ("college", "kolled", "tehnik", "tehnic", "lyceum", "ssuz", "spo", "tekhnikum")

def clean_url_one(u: str,
                  drop_ssuz: bool = False,
                  drop_k12: bool = False,
                  collapse_lang_adm: bool = True) -> str | None:
    if not u:
        return None
    s = urlsplit(str(u).strip())
    host = (s.netloc or s.path).split("/")[0].lower()
    if not host or host.startswith(("mailto:", "javascript:")):
        return None
    if host.startswith("www."):
        host = host[4:]
    host = host.split(":", 1)[0]

    # IDN → punycode (canonical)
    try:
        host = host.encode("idna").decode("ascii")
    except Exception:
        pass

    ext = tldextract.extract(host)  # subdomain, domain, suffix
    if not ext.suffix or not ext.domain:
        return None
    etld1 = f"{ext.domain}.{ext.suffix}"
    sub = ext.subdomain

    # Drop known aggregators/social by registrable domain
    if etld1 in HOST_DROP_ETLD1:
        return None
    # Drop ONLY the edu.ru portal and its true subdomains
    if ext.domain == "edu" and ext.suffix == "ru":
        return None

    # Optional filters
    if drop_ssuz and any(h in host for h in SSUZ_HINTS):
        return None
    if drop_k12 and (".mskobr.ru" in host or host.endswith(".eduru.ru") or ".obr." in host):
        return None

    # Collapse language/admissions subdomains (but keep institutional subs like academy.customs.gov.ru)
    if collapse_lang_adm and sub in COLLAPSE_PREFIXES:
        host = etld1

    return f"https://{host}/"

assert clean_url_one("https://bsuedu.ru/") == "https://bsuedu.ru/"
assert clean_url_one("https://sfedu.ru/")  == "https://sfedu.ru/"
assert clean_url_one("https://asu-edu.ru/")== "https://asu-edu.ru/"
assert clean_url_one("https://www.edu.ru/vuz/card/...") is None
assert clean_url_one("https://ru.wikipedia.org/...",) is None

# Collapse admissions/lang subs but keep institutional subs
assert clean_url_one("https://pk.aumsu.ru/").endswith("aumsu.ru/")
assert clean_url_one("https://abitur.penzgtu.ru/").endswith("penzgtu.ru/")
assert clean_url_one("https://academy.customs.gov.ru/") == "https://academy.customs.gov.ru/"

# Optional K-12 drop
assert clean_url_one("https://madk.mskobr.ru/", drop_k12=True) is None

universities_df['clean_url'] = universities_df['url'].apply(lambda u: clean_url_one(u, drop_ssuz=False))

print(f"After cleaning, {universities_df['clean_url'].notna().sum()} / {len(universities_df)} universities have a valid URL.")

#%%
# find duplicate cleaned URLs and keep only the row with the most students for each duplicate
clean_series = universities_df['clean_url'].dropna()
counts = clean_series.value_counts()
dup_urls = counts[counts > 1].index.tolist()

print(f"Total rows with clean_url: {len(clean_series)}")
print(f"Unique clean_url values: {len(counts)}")
print(f"Clean URLs that are duplicated: {len(dup_urls)} (affecting {int(counts[counts > 1].sum())} rows)")

affected = 0
for url in dup_urls:
    group_idx = universities_df[universities_df['clean_url'] == url].index
    # choose index with max students_amount to keep
    keep_idx = universities_df.loc[group_idx, 'students_amount'].idxmax()
    # all other indices for this url -> set clean_url to None
    drop_idx = [i for i in group_idx if i != keep_idx]
    if drop_idx:
        universities_df.loc[drop_idx, 'clean_url'] = None
        affected += len(drop_idx)

# update counts/flags
clean_series = universities_df['clean_url'].dropna()
counts = clean_series.value_counts()
universities_df['clean_url_count'] = universities_df['clean_url'].map(counts).fillna(0).astype(int)
universities_df['clean_url_is_dup'] = universities_df['clean_url_count'] > 1

print(f"Set clean_url=None for {affected} duplicate rows. Remaining duplicated clean_url groups: {len(counts[counts > 1])}")
# show remaining duplicates if any
if (counts > 1).any():
    dup_rows = universities_df[universities_df['clean_url_is_dup']].sort_values(['clean_url', 'name'])
    display_cols = ['clean_url', 'name', 'abbreviation', 'students_amount', 'teachers_amount', 'url', 'clean_url_count']
    print(dup_rows[display_cols].to_string(index=False))

#%%
universities_df.to_csv('data/generated/universities_cleaned.csv', index=False)