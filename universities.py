#%%
import src.google_search as google_search
import pandas as pd
from tqdm import tqdm

#%%
universities_df = pd.read_csv('data/download/partners.csv')
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
print(f"Loaded {len(universities_df)} rows.")
# ensure students_amount and teachers_amount are ints, convert NaNs to 0, drop rows where either is 0
for col in ['students_amount', 'teachers_amount']:
    universities_df[col] = pd.to_numeric(universities_df[col], errors='coerce').fillna(0).astype(int)

universities_df = universities_df[(universities_df['students_amount'] >= 100) & (universities_df['teachers_amount'] >= 10)].copy().reset_index(drop=True)
print(f"After filtering small universities, {len(universities_df)} rows remain.")

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
    "youtube.com", "t.me", "vuzopedia.ru", "postupi.online", "universitys.ru", "ucheba.ru", "ivobr.ru", "ciur.ru", "robogeek.ru", "cdnstatic.rg.ru", "tabiturient.ru", "vuzopedia.ru", "abiturient.ru", "vuzopedia.ru", "studopedia.ru", "studfile.net", "studbooks.net", "studme.org", "studref.com", "referat911.ru", "referat.guru", "allbest.ru", "znanium.com", "libgen.is", "libgen.rs", "libgen.li", "e-lib.info", "academia.edu", "researchgate.net", "akkork.ru", "i-exam.ru", "minobr63.ru", "mskobr.ru", "dagestanschool.ru"
}
COLLAPSE_PREFIXES = {"en", "ru", "pk", "abitur", "priem", "admission", "admit", "new"}
SSUZ_HINTS = ("college", "kolled", "tehnik", "tehnic", "lyceum", "ssuz", "spo", "tekhnikum")

def clean_url_one(u: str,
                  drop_ssuz: bool = False,
                  collapse_lang_adm: bool = True) -> str | None:
    if not u:
        return None
    s = urlsplit(str(u).strip())
    host = (s.netloc or s.path).split("/")[0].lower()
    if not host or host.startswith(("mailto:", "javascript:")):
        return None
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

    # Collapse language/admissions subdomains (but keep institutional subs like academy.customs.gov.ru)
    if collapse_lang_adm and sub in COLLAPSE_PREFIXES:
        host = etld1

    # preserve original http if present, otherwise default to https
    scheme = s.scheme.lower()
    if scheme != "http":
        scheme = "https"

    return f"{scheme}://{host}/"

assert clean_url_one("https://bsuedu.ru/") == "https://bsuedu.ru/"
assert clean_url_one("https://sfedu.ru/")  == "https://sfedu.ru/"
assert clean_url_one("https://asu-edu.ru/")== "https://asu-edu.ru/"
assert clean_url_one("https://www.edu.ru/vuz/card/...") is None
assert clean_url_one("https://ru.wikipedia.org/...",) is None

# Collapse admissions/lang subs but keep institutional subs
assert clean_url_one("https://pk.aumsu.ru/").endswith("aumsu.ru/")
assert clean_url_one("https://abitur.penzgtu.ru/").endswith("penzgtu.ru/")
assert clean_url_one("https://academy.customs.gov.ru/") == "https://academy.customs.gov.ru/"

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
before = len(universities_df)
mask = universities_df['clean_url'].notna() & (universities_df['clean_url'].astype(str).str.strip() != "")
universities_df = universities_df[mask].copy().reset_index(drop=True)
after = len(universities_df)
print(f"Dropped {before - after} rows without clean_url; {after} rows remain.")

#%%
import src.url_utils as url_utils
universities_df['url_root'] = universities_df['clean_url'].apply(lambda u: url_utils.extract_root(u) if pd.notna(u) else None)
print(f"After extracting url_root, {universities_df['url_root'].notna().sum()} / {len(universities_df)} universities have a valid url_root.")
print(f"Unique url_root values: {universities_df['url_root'].nunique()}")

#%%
# show duplicate url_root values and affected rows, then keep only the university with most students per root
root_series = universities_df['url_root'].dropna()
counts = root_series.value_counts()
dup_roots = counts[counts > 1].index.tolist()

print(f"Total rows with url_root: {len(root_series)}")
print(f"Unique url_root values: {len(counts)}")
print(f"Duplicate url_root values: {len(dup_roots)} (affecting {int(counts[counts > 1].sum())} rows)")

if dup_roots:
    dup_rows = universities_df[universities_df['url_root'].isin(dup_roots)].sort_values(['url_root', 'name'])
    display_cols = ['url_root', 'name', 'abbreviation', 'students_amount', 'teachers_amount', 'url']
    # print(dup_rows[display_cols].to_string(index=False))

    # For each duplicated root, keep only the row with the most students; set others' url_root to None
    affected = 0
    for root in dup_roots:
        group_idx = universities_df[universities_df['url_root'] == root].index.tolist()
        keep_idx = universities_df.loc[group_idx, 'students_amount'].idxmax()
        drop_idx = [i for i in group_idx if i != keep_idx]
        if drop_idx:
            universities_df.loc[drop_idx, 'url_root'] = None
            affected += len(drop_idx)

    print(f"Set url_root=None for {affected} duplicate rows (kept the university with most students per root).")

    # show any remaining duplicates (should be none)
    root_series = universities_df['url_root'].dropna()
    counts = root_series.value_counts()
    remaining_dup_roots = counts[counts > 1].index.tolist()
    print(f"Remaining duplicate url_root groups: {len(remaining_dup_roots)}")
    if remaining_dup_roots:
        dup_rows = universities_df[universities_df['url_root'].isin(remaining_dup_roots)].sort_values(['url_root', 'name'])
        print(dup_rows[display_cols].to_string(index=False))
else:
    print("No duplicate url_root values found.")

#%%
universities_df.to_csv('data/generated/universities_cleaned.csv', index=False)
# %%
