#%%
from urllib.parse import urlparse
import tldextract  # pip install tldextract


def extract_root(url: str) -> str:
    """Return the logical root identifier for a university / institution URL.

    Heuristics (expanded as test set grows):
    1. Normal case: return registrable domain (PSL-based).
    2. Aggregator paths on *.edu.ru: /vuz/card/<slug>/... -> slug.
    3. Provider / regional second-level domains (e.g. spb.ru, 1mcg.ru, 68edu.ru,
       dagestanschool.ru, perm.ru, nnov.ru, rosguard.gov.ru, sakhalin.gov.ru, etc.)
       represent a namespace where the *first* subdomain label (after removing
       generic prefixes like www) is the institution id.
    4. For gov.ru trees we usually take the first subdomain label unless the
       first label is a generic container like 'academy' (then keep domain).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return ""

    # 2. edu.ru institution cards
    if host.endswith("edu.ru"):
        path_parts = [p for p in parsed.path.split('/') if p]
        for i in range(len(path_parts) - 2):
            if path_parts[i] == 'vuz' and path_parts[i + 1] == 'card':
                return path_parts[i + 2]

    ext = tldextract.extract(host)

    # 3. Domains where first subdomain label is the meaningful org code
    FIRST_LABEL_DOMAINS = {
        # city / region style
        'spb', 'obninsk', 'samregion', 'ac',
        # provider-style / educational hosting platforms
        '1mcg', '3dn', '68edu', 'edu35', 'dagestanschool', 'edu22', 'educrimea', 'edusev',
        # regional education / institute platforms
        'irk', 'iro61', 'kemobl', 'perm', 'nnov',
        # government hierarchies / ministerial parents (except some generic subdomain containers)
        'rosguard', 'sakhalin', 'sakha', 'mil', 'minobr63'
    }
    GENERIC_SUBDOMAIN_PREFIXES = {'www'}
    GENERIC_CONTAINER_SUBDOMAINS = {'academy'}  # if sole subdomain before domain

    # If the effective 'domain' component (per tldextract) is itself in our list treat the first subdomain label
    # as institution identifier (after stripping generic prefixes).
    if (ext.domain in FIRST_LABEL_DOMAINS) and ext.subdomain:
        labels = [l for l in ext.subdomain.split('.') if l]
        # Drop generic leading prefixes
        while labels and labels[0] in GENERIC_SUBDOMAIN_PREFIXES:
            labels = labels[1:]
        # If only one label and it's a generic container (e.g. academy.customs.gov.ru case handled elsewhere)
        if labels and labels[0] in GENERIC_CONTAINER_SUBDOMAINS and len(labels) == 1:
            # fall back to domain name (e.g. keep 'customs')
            pass
        else:
            if labels:
                return labels[0]

    # 4. Hierarchical gov.ru handling: tldextract treats 'gov.ru' as domain=something? Actually domain=='gov', suffix=='ru'.
    # Pattern: <unit>.<parent>.gov.ru OR <unit>.<dept>.<region>.gov.ru etc. We want the first label unless it's a generic
    # container like 'academy' where we instead take the next label.
    if ext.domain == 'gov' and ext.suffix == 'ru' and ext.subdomain:
        labels = [l for l in ext.subdomain.split('.') if l]
        # Remove generic www
        while labels and labels[0] in GENERIC_SUBDOMAIN_PREFIXES:
            labels = labels[1:]
        if not labels:
            return 'gov'
        # academy.<something>.gov.ru -> something
        if labels[0] in GENERIC_CONTAINER_SUBDOMAINS and len(labels) >= 2:
            return labels[1]
        # default: first label
        return labels[0]

    # 5. Commercial style second-levels: *.com.ru, *.net.ru
    if ext.domain in {'com', 'net'} and ext.suffix == 'ru' and ext.subdomain:
        labels = [l for l in ext.subdomain.split('.') if l]
        while labels and labels[0] in GENERIC_SUBDOMAIN_PREFIXES:
            labels = labels[1:]
        if labels:
            return labels[0]

    # 1. default
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

# new 4
assert extract_root("https://mpei.ru/sveden/document/Documents/report_2023s.pdf") == "mpei", extract_root("https://mpei.ru/sveden/document/Documents/report_2023s.pdf")
assert extract_root("http://ispu.ru/files/u2/sveden/education/RPD_01.03.02_02_O_3.pdf") == "ispu", extract_root("http://ispu.ru/files/u2/sveden/education/RPD_01.03.02_02_O_3.pdf")


# new 5
assert extract_root("http://npy.1mcg.ru/") == "npy", extract_root("http://npy.1mcg.ru/")
assert extract_root("https://tkt.3dn.ru/") == "tkt", extract_root("https://tkt.3dn.ru/")
assert extract_root("https://agrotteh.68edu.ru/") == "agrotteh", extract_root("https://agrotteh.68edu.ru/")
assert extract_root("https://lesmeh.edu35.ru/") == "lesmeh", extract_root("https://lesmeh.edu35.ru/")

# 2) “city-portal / academic” style second-level hosts (need exceptions like 'spb.ru', 'ac.ru')
assert extract_root("http://www.vspu.ac.ru/") == "vspu", extract_root("http://www.vspu.ac.ru/")
assert extract_root("http://ivanovo.ac.ru/") == "ivanovo", extract_root("http://ivanovo.ac.ru/")
assert extract_root("https://www.uniyar.ac.ru/") == "uniyar", extract_root("https://www.uniyar.ac.ru/")

# 3) multi-tenant provider domains
assert extract_root("https://pedcollege-derbent.dagestanschool.ru/") == "pedcollege-derbent", extract_root("https://pedcollege-derbent.dagestanschool.ru/")
assert extract_root("https://bgpk.edu22.info/") == "bgpk", extract_root("https://bgpk.edu22.info/")
assert extract_root("https://akpf.educrimea.ru/") == "akpf", extract_root("https://akpf.educrimea.ru/")
assert extract_root("https://sevask.edusev.ru/") == "sevask", extract_root("https://sevask.edusev.ru/")

# 4) regional/departmental “*.gov.ru” trees (PSL makes 'gov.ru' a suffix)
assert extract_root("https://nvi.rosguard.gov.ru/") == "nvi", extract_root("https://nvi.rosguard.gov.ru/")
assert extract_root("https://academy.customs.gov.ru/") == "customs", extract_root("https://academy.customs.gov.ru/")
assert extract_root("https://sakhsjh.sakhalin.gov.ru/") == "sakhsjh", extract_root("https://sakhsjh.sakhalin.gov.ru/")
assert extract_root("https://ykt-yaksit.obr.sakha.gov.ru/") == "ykt-yaksit", extract_root("https://ykt-yaksit.obr.sakha.gov.ru/")

# 5) other regional second-levels
assert extract_root("http://itam.irk.ru/") == "itam", extract_root("http://itam.irk.ru/")
assert extract_root("https://kkpt-sulin.iro61.ru/") == "kkpt-sulin", extract_root("https://kkpt-sulin.iro61.ru/")
assert extract_root("https://kptt.kemobl.ru/") == "kptt", extract_root("https://kptt.kemobl.ru/")
assert extract_root("https://prk.perm.ru/") == "prk", extract_root("https://prk.perm.ru/")
assert extract_root("https://netk.nnov.ru/") == "netk", extract_root("https://netk.nnov.ru/")

# 6) special SLDs
assert extract_root("https://ahtt.com.ru/") == "ahtt", extract_root("https://ahtt.com.ru/")
assert extract_root("https://vpk.net.ru/") == "vpk", extract_root("https://vpk.net.ru/")

# 7) military/ministerial hosts
assert extract_root("https://varhbz.mil.ru/") == "varhbz", extract_root("https://varhbz.mil.ru/")
assert extract_root("https://po.minobr63.ru/") == "po", extract_root("https://po.minobr63.ru/")

# %%
