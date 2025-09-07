#%%
import pandas as pd
import re
from urllib.parse import urlparse

#%%
from urllib.parse import urlparse

_THROWAWAY_SUBS = {"www","m","new","old","beta","dev","portal","web","web-edu","pk","po","nti"}
# generic SLDs frequently used under ccTLDs (e.g., edu.kz, ac.uk, com.ru, etc.)
_GENERIC_SLD = {"edu","ac","gov","mil","com","net","org","co"}

def _domain_root(labels):
    n = len(labels)
    if n == 1:
        return labels[0]
    # If ccTLD (last label length==2) and second-to-last is a generic SLD → take label before it
    if n >= 3 and len(labels[-1]) == 2 and labels[-2] in _GENERIC_SLD:
        return labels[-3]
    # Default: take SLD
    return labels[-2]

def extract_root(url: str, decode_punycode: bool = False) -> str:
    p = urlparse(url)
    host = (p.netloc or url).split("@")[-1].split(":")[0].lower().strip(".")
    path = p.path or "/"

    # Special handling for federal portal *.edu.ru → slug in /vuz/card/<slug>/
    if host.endswith(".edu.ru") or host == "edu.ru":
        parts = [s for s in path.strip("/").split("/") if s]
        if len(parts) >= 3 and parts[0] == "vuz" and parts[1] == "card":
            return parts[2]

    labels = [s for s in host.split(".") if s]
    while len(labels) >= 3 and labels[0] in _THROWAWAY_SUBS:
        labels.pop(0)

    root = _domain_root(labels) if labels else ""
    if decode_punycode and root:
        try:
            root = root.encode("ascii").decode("idna")
        except Exception:
            pass
    return root

# --- tests ---
assert extract_root("https://ccu.edu.kz/") == "ccu"
assert extract_root("https://www.hse.ru/") == "hse"
assert extract_root("https://spbu.ru/") == "spbu"
assert extract_root("http://ispu.ru/files/u2/sveden/education/doc.pdf") == "ispu"
assert extract_root("https://kpfu.ru/portal/docs/file.pdf") == "kpfu"

assert extract_root("https://new.mosap.ru/") == "mosap"
assert extract_root("http://www.sibsau.ru/page/home/") == "sibsau"
assert extract_root("https://web-edu.rsreu.ru/res/programs-file-storage/abc.pdf") == "rsreu"

assert extract_root("https://edu.tatar.ru/sovetcki/org6264/page2474457.htm") == "tatar"

assert extract_root("https://www.edu.ru/vuz/card/institut-zakonovedeniya-i-upravleniya-vserossijskoj-policejskoj-associacii/contacts") \
       == "institut-zakonovedeniya-i-upravleniya-vserossijskoj-policejskoj-associacii"

assert extract_root("https://xn--80af5bzc.xn--p1ai/") == "xn--80af5bzc"

print("All tests passed.")

#%%
df = pd.read_csv("data/generated/specialities_with_study_plans.csv", delimiter=";")
university_df = pd.read_csv("data/generated/universities.csv")
university_df['url'] = university_df['url'].apply(lambda u: "Unknown" if 'wiki' in str(u).lower() else u)

university_df['url_root'] = university_df['url'].apply(extract_root)

#%%
for (idx, row) in df.iloc[20:40].iterrows():
    print(f"[{idx}] {row['speciality_name']} ({row['speciality_code']}): {row['study_plan_url']}")
    url = row['study_plan_url']

    uni = "Unknown"
    matches = []
    if not url or pd.isna(url):
        print(f"study_plan_url is empty or NaN for idx={idx}")
    else:
        for _, urow in university_df.iterrows():
            root = urow.get("url_root")
            if not root or root == "unknown":
                continue
            if re.search(rf'\b{re.escape(str(root))}\b', str(url), re.IGNORECASE):
                matches.append((urow['abbreviation'], root, urow['url']))

    if len(matches) == 0:
        print(f"No matching university found for study_plan url={url}")
    elif len(matches) > 1:
        print(f"Multiple ({len(matches)}) matching universities for study_plan url={url}:")
        for abbr, root, uurl in matches:
            print(f" - {abbr}, root={root}, url={uurl}")
    else:
        uni, uni_root, uni_url = matches[0]
        print(f"Matching university: {uni}, url={uni_url}, root={uni_root}")
# %%
