#%%
from urllib.parse import urlparse
import tldextract  # pip install tldextract


def extract_root(url: str) -> str:
    """Return the logical root identifier for a university / institution URL.

    Rules inferred from tests:
    1. For ordinary domains return the registrable domain (e.g. kpfu.ru -> kpfu).
    2. If host is a regional / generic second level like *.spb.ru we want the
       left-most subdomain label instead (miep.spb.ru -> miep).
       (Currently only 'spb' is required by the tests; easy to extend.)
    3. For URLs on *.edu.ru of the form /vuz/card/<slug>/... return that <slug>
       instead of the host's domain (the site aggregates many institutions).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return ""

    # Special path-based extraction for aggregator site edu.ru
    if host.endswith("edu.ru"):
        # Look for /vuz/card/<slug>/ pattern in the path
        path_parts = [p for p in parsed.path.split('/') if p]
        for i in range(len(path_parts) - 2):
            if path_parts[i] == 'vuz' and path_parts[i + 1] == 'card':
                return path_parts[i + 2]
        # fall back to normal domain extraction if pattern not found

    ext = tldextract.extract(host)  # uses Public Suffix List

    # Regional generic second-level domains where subdomain is institution
    GENERIC_SECOND_LEVEL = {"spb"}
    if ext.domain in GENERIC_SECOND_LEVEL and ext.subdomain:
        labels = [l for l in ext.subdomain.split('.') if l]
        # Drop leading 'www' if present
        if labels and labels[0] == 'www':
            labels = labels[1:]
        return labels[0] if labels else ext.subdomain.split('.')[0]

    return ext.domain or (host.split('.')[0] if host else "")

# --- tests ---
assert extract_root("https://miep.spb.ru/") == "miep", extract_root("https://miep.spb.ru/")
assert extract_root("https://ccu.edu.kz/") == "ccu", extract_root("https://ccu.edu.kz/")
assert extract_root("https://www.hse.ru/") == "hse", extract_root("https://www.hse.ru/")
assert extract_root("https://spbu.ru/") == "spbu", extract_root("https://spbu.ru/")
assert extract_root("http://ispu.ru/files/u2/sveden/education/doc.pdf") == "ispu", extract_root("http://ispu.ru/files/u2/sveden/education/doc.pdf")
assert extract_root("https://kpfu.ru/portal/docs/file.pdf") == "kpfu", extract_root("https://kpfu.ru/portal/docs/file.pdf")

assert extract_root("https://new.mosap.ru/") == "mosap", extract_root("https://new.mosap.ru/")
assert extract_root("http://www.sibsau.ru/page/home/") == "sibsau", extract_root("http://www.sibsau.ru/page/home/")
assert extract_root("https://web-edu.rsreu.ru/res/programs-file-storage/abc.pdf") == "rsreu", extract_root("https://web-edu.rsreu.ru/res/programs-file-storage/abc.pdf")

assert extract_root("https://edu.tatar.ru/sovetcki/org6264/page2474457.htm") == "tatar", extract_root("https://edu.tatar.ru/sovetcki/org6264/page2474457.htm")

assert extract_root("https://www.edu.ru/vuz/card/institut-zakonovedeniya-i-upravleniya-vserossijskoj-policejskoj-associacii/contacts") \
       == "institut-zakonovedeniya-i-upravleniya-vserossijskoj-policejskoj-associacii", extract_root("https://www.edu.ru/vuz/card/institut-zakonovedeniya-i-upravleniya-vserossijskoj-policejskoj-associacii/contacts")

# new
assert extract_root("https://apmath.spbu.ru/images/Documenty/Programs-2024.pdf") == "spbu", extract_root("https://apmath.spbu.ru/images/Documenty/Programs-2024.pdf")
assert extract_root("https://mpei.ru/sveden/document/Documents/report_2023s.pdf") == "mpei", extract_root("https://mpei.ru/sveden/document/Documents/report_2023s.pdf")
assert extract_root("https://download.guap.ru/sveden/6049/rpd_gosudarstvennaya_itogovaya_attestaciya_220250.pdf") == "guap", extract_root("https://download.guap.ru/sveden/6049/rpd_gosudarstvennaya_itogovaya_attestaciya_220250.pdf")
assert extract_root("https://mpei.ru/sveden/document/Documents/report_2022.pdf") == "mpei", extract_root("https://mpei.ru/sveden/document/Documents/report_2022.pdf")
assert extract_root("http://www.mivlgu.ru/site_arch/documents/akkred/2015/otchet_samoobsled_10.03.01-n.pdf") == "mivlgu", extract_root("http://www.mivlgu.ru/site_arch/documents/akkred/2015/otchet_samoobsled_10.03.01-n.pdf")
assert extract_root("https://computer.susu.ru/_images/for-entrant.pdf") == "susu", extract_root("https://computer.susu.ru/_images/for-entrant.pdf")
assert extract_root("https://programms.edu.urfu.ru/ru/9986/documents/") == "urfu", extract_root("https://programms.edu.urfu.ru/ru/9986/documents/")
assert extract_root("https://math-cs.spbu.ru/wp-content/uploads/2024/03/Elektivy-4-kurs-20_5152.pdf") == "spbu", extract_root("https://math-cs.spbu.ru/wp-content/uploads/2024/03/Elektivy-4-kurs-20_5152.pdf")
assert extract_root("https://fgosvo.ru/uploadfiles/Projects_POOP/BAK/020301_POOP_B.pdf") == "fgosvo", extract_root("https://fgosvo.ru/uploadfiles/Projects_POOP/BAK/020301_POOP_B.pdf")
assert extract_root("http://web.ugatu.su/assets/files/documents/study/uplan/bakal/02.03.01_MiKN-MiKM/OOP_02.03.01_MiKN-MiKM_29.05.2015.pdf") == "ugatu", extract_root("http://web.ugatu.su/assets/files/documents/study/uplan/bakal/02.03.01_MiKN-MiKM/OOP_02.03.01_MiKN-MiKM_29.05.2015.pdf")
assert extract_root("https://web-edu.rsreu.ru/res/programs-file-storage/37dc27bcf6fef3d5.pdf") == "rsreu", extract_root("https://web-edu.rsreu.ru/res/programs-file-storage/37dc27bcf6fef3d5.pdf")
assert extract_root("https://cs.msu.ru/sites/cmc/files/docs/uchplan_fiit_2023.pdf") == "msu", extract_root("https://cs.msu.ru/sites/cmc/files/docs/uchplan_fiit_2023.pdf")
assert extract_root("https://programs.edu.urfu.ru/media/rpm/00031688.pdf") == "urfu", extract_root("https://programs.edu.urfu.ru/media/rpm/00031688.pdf")
assert extract_root("https://f.physchem.msu.ru/docs/education_programs/%D0%98%D0%91_%D0%9F%D1%80%D0%B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D1%8B%D0%B5%20%D0%BC%D0%B0%D1%82%D0%B5%D0%BC%D0%B0%D1%82%D0%B8%D0%BA%D0%B0%20%D0%B8%20%D1%84%D0%B8%D0%B7%D0%B8%D0%BA%D0%B0_%D0%A4%D0%A4%D0%A5%D0%98_2023.pdf") == "msu", extract_root("https://f.physchem.msu.ru/docs/education_programs/%D0%98%D0%91_%D0%9F%D1%80%D0%B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D1%8B%D0%B5%20%D0%BC%D0%B0%D1%82%D0%B5%D0%BC%D0%B0%D1%82%D0%B8%D0%BA%D0%B0%20%D0%B8%20%D1%84%D0%B8%D0%B7%D0%B8%D0%BA%D0%B0_%D0%A4%D0%A4%D0%A5%D0%98_2023.pdf")

# new 2
assert extract_root("https://miep.spb.ru/") == "miep", extract_root("https://miep.spb.ru/")
assert extract_root("https://herzen.spb.ru/") == "herzen", extract_root("https://herzen.spb.ru/")

# Control cases show the default PSL behavior still works:
assert extract_root("https://apmath.spbu.ru/") == "spbu", extract_root("https://apmath.spbu.ru/")
assert extract_root("https://math-cs.spbu.ru/wp/...") == "spbu", extract_root("https://math-cs.spbu.ru/wp/...")
assert extract_root("https://cs.msu.ru/") == "msu", extract_root("https://cs.msu.ru/")
assert extract_root("https://web-edu.rsreu.ru/res/...") == "rsreu", extract_root("https://web-edu.rsreu.ru/res/...")
assert extract_root("https://ccu.edu.kz/") == "ccu", extract_root("https://ccu.edu.kz/")
assert extract_root("https://new.mosap.ru/") == "mosap", extract_root("https://new.mosap.ru/")

# new 3
assert extract_root("https://www.ranepa.ru/") == "ranepa", extract_root("https://www.ranepa.ru/")
assert extract_root("https://uust.ru/") == "uust", extract_root("https://uust.ru/")
assert extract_root("https://synergy.ru/") == "synergy", extract_root("https://synergy.ru/")
assert extract_root("https://www.fa.ru/") == "fa", extract_root("https://www.fa.ru/")
assert extract_root("https://urfu.ru/ru/") == "urfu", extract_root("https://urfu.ru/ru/")
assert extract_root("https://donstu.ru/") == "donstu", extract_root("https://donstu.ru/")
assert extract_root("https://www.rea.ru/") == "rea", extract_root("https://www.rea.ru/")
assert extract_root("https://www.mfua.ru/") == "mfua", extract_root("https://www.mfua.ru/")
assert extract_root("https://www.rudn.ru/") == "rudn", extract_root("https://www.rudn.ru/")
assert extract_root("https://www.ruc.su/") == "ruc", extract_root("https://www.ruc.su/")
assert extract_root("https://www.spbstu.ru/") == "spbstu", extract_root("https://www.spbstu.ru/")
assert extract_root("http://www.miit.ru/") == "miit", extract_root("http://www.miit.ru/")
assert extract_root("https://www.kubsu.ru/") == "kubsu", extract_root("https://www.kubsu.ru/")
assert extract_root("https://bmstu.ru/") == "bmstu", extract_root("https://bmstu.ru/")
assert extract_root("https://mi.university/") == "mi", extract_root("https://mi.university/")
assert extract_root("https://rpa-mu.ru/") == "rpa-mu", extract_root("https://rpa-mu.ru/")
assert extract_root("https://www.sfu-kras.ru/") == "sfu-kras", extract_root("https://www.sfu-kras.ru/")
assert extract_root("https://www.herzen.spb.ru/") == "herzen", extract_root("https://www.herzen.spb.ru/")
assert extract_root("https://mephi.ru/node") == "mephi", extract_root("https://mephi.ru/node")
assert extract_root("https://www.pushkin.institute/") == "pushkin", extract_root("https://www.pushkin.institute/")
assert extract_root("https://mti.moscow/") == "mti", extract_root("https://mti.moscow/")
assert extract_root("https://xn--80af5bzc.xn--p1ai/ru/") == "xn--80af5bzc", extract_root("https://xn--80af5bzc.xn--p1ai/ru/")

# %%
