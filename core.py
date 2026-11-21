"""Logika data & GMM (tanpa Streamlit/Plotly) agar bisa diuji sendiri: python core.py"""
import glob
import io
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 42
CODE = "SL.TLF.CACT.FE.ZS"
YEARS = list(range(1960, 2024))
FEATURES = ["FLPR_2000", "FLPR_2010", "FLPR_2020", "FLPR_2023", "Mean_FLPR", "Growth_Rate", "Std_Dev_FLPR"]
BORDER_LO, BORDER_HI = 0.35, 0.65

PROFILES = {
    "high": ("High & Stable", "Partisipasi tinggi dan stabil.",
             "Jadikan benchmark regulasi cuti melahirkan berbayar dan insentif kesetaraan upah; pertahankan dengan pemantauan kesenjangan upah gender."),
    "low": ("Stagnant / Borderline Low", "Partisipasi rendah dengan pertumbuhan terbatas.",
            "Lakukan pay equity audit, beri insentif pajak WFH/jam kerja fleksibel, dan subsidi daycare untuk menurunkan hambatan masuk kerja."),
    "trans": ("Rapid Transition / Emerging Growth", "Partisipasi menengah dan sedang bertransisi.",
              "Sediakan kredit mikro khusus pengusaha perempuan serta pelatihan literasi keuangan dan digital."),
}


# ---------- Pemuatan data ----------
def _wide(values, meta):
    """values: DataFrame [Country Code, Country Name, tahun...]; meta: [Country Code, Region, IncomeGroup]."""
    meta = meta[meta["Region"].notna() & (meta["Region"] != "Aggregates")]
    w = values.merge(meta[["Country Code", "Region", "IncomeGroup"]], on="Country Code")
    w["IncomeGroup"] = w["IncomeGroup"].fillna("Tidak diklasifikasi")
    for y in YEARS:
        if y not in w.columns:
            w[y] = np.nan
    return w.set_index("Country Code")[["Country Name", "Region", "IncomeGroup"] + YEARS]


def load_csv(api_csv, meta_csv):
    """Berkas CSV World Bank (path atau file-like)."""
    v = pd.read_csv(api_csv, skiprows=4, encoding="utf-8-sig")
    if hasattr(meta_csv, "seek"):
        meta_csv.seek(0)
    m = pd.read_csv(meta_csv, encoding="utf-8-sig")
    v = v.rename(columns={c: int(c) for c in v.columns if str(c).isdigit()})
    return _wide(v[["Country Name", "Country Code"] + [c for c in v.columns if isinstance(c, int)]], m)


def load_local(folder):
    a = glob.glob(os.path.join(folder, "API_SL*CACT*FE*ZS*.csv"))
    m = glob.glob(os.path.join(folder, "Metadata_Country_API_SL*CACT*FE*ZS*.csv"))
    if not (a and m):
        raise FileNotFoundError("Berkas CSV lokal tidak ditemukan")
    return load_csv(a[0], m[0])


def load_api(timeout=10):
    import requests
    base = "https://api.worldbank.org/v2"
    rows, page = [], 1
    while True:
        r = requests.get(f"{base}/country/all/indicator/{CODE}", timeout=timeout,
                         params=dict(format="json", per_page=20000, date="1960:2023", page=page))
        r.raise_for_status()
        j = r.json()
        if len(j) < 2 or j[1] is None:
            raise RuntimeError("Respons API World Bank tidak valid")
        rows += [(d["countryiso3code"], d["country"]["value"], int(d["date"]), d["value"]) for d in j[1]]
        if page >= j[0]["pages"]:
            break
        page += 1
    long = pd.DataFrame(rows, columns=["Country Code", "Country Name", "year", "value"])
    v = long.pivot_table(index=["Country Code", "Country Name"], columns="year", values="value").reset_index()
    c = requests.get(f"{base}/country", params=dict(format="json", per_page=400), timeout=timeout)
    c.raise_for_status()
    m = pd.DataFrame([(d["id"], d["region"]["value"], d["incomeLevel"]["value"]) for d in c.json()[1]],
                     columns=["Country Code", "Region", "IncomeGroup"])
    return _wide(v, m)


def synthetic(n=150):
    """Data sintetis realistis (tiga arketipe) bila API/CSV tidak tersedia."""
    rng = np.random.default_rng(SEED)
    regions = ["Europe & Central Asia", "Sub-Saharan Africa", "Latin America & Caribbean", "East Asia & Pacific",
               "Middle East & North Africa", "South Asia", "North America"]
    incomes = ["High income", "Upper middle income", "Lower middle income", "Low income"]
    t = np.arange(1960, 2024) - 1990
    out = []
    for i in range(n):
        a = rng.choice(3, p=[0.35, 0.35, 0.30])
        if a == 0:    # tinggi & stabil
            y = rng.normal(62, 7) + rng.normal(0, 0.05) * t
        elif a == 1:  # rendah / stagnan
            y = rng.normal(27, 6) + rng.normal(0.05, 0.05) * t
        else:         # transisi cepat
            y = rng.normal(35, 5) + rng.normal(0.45, 0.1) * t
        y = np.clip(y + rng.normal(0, 0.6, len(t)).cumsum() * 0.15 + rng.normal(0, 0.5, len(t)), 5, 90)
        y[t < 0] = np.nan
        y[rng.random(len(y)) < 0.03] = np.nan
        out.append([f"S{i:03d}", f"Negara Sintetis {i+1:03d}", rng.choice(regions), rng.choice(incomes, p=[.3, .3, .25, .15]), *y])
    return pd.DataFrame(out, columns=["Country Code", "Country Name", "Region", "IncomeGroup"] + YEARS).set_index("Country Code")


# ---------- Rekayasa fitur & preprocessing ----------
def build_features(w):
    last = w[list(range(2014, 2024))]
    f = pd.DataFrame(index=w.index)
    for y in (2000, 2010, 2020, 2023):
        f[f"FLPR_{y}"] = w[y]
    f["Mean_FLPR"] = last.mean(axis=1)
    f["Growth_Rate"] = w[2023] - w[2013]
    f["Std_Dev_FLPR"] = last.std(axis=1)
    return f[f.isna().sum(axis=1) <= 3]  # buang negara dengan data nyaris kosong


def preprocess(f):
    """Imputasi multivariat (BayesianRidge) -> winsorisasi IQR x3 -> Z-score."""
    imp = pd.DataFrame(
        IterativeImputer(estimator=BayesianRidge(), max_iter=25, random_state=SEED).fit_transform(f),
        index=f.index, columns=f.columns)
    q1, q3 = imp.quantile(0.25), imp.quantile(0.75)
    lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    clipped = imp.clip(lo, hi, axis=1)
    n_clipped = int((clipped != imp).sum().sum())
    sc = StandardScaler().fit(clipped)
    scaled = pd.DataFrame(sc.transform(clipped), index=f.index, columns=f.columns)
    return dict(raw=f, imputed=imp, clean=clipped, scaled=scaled, scaler=sc, n_clipped=n_clipped)


# ---------- GMM ----------
def fit_gmm(X, k, reg=0.05):
    return GaussianMixture(k, covariance_type="full", init_params="kmeans", n_init=10,
                           reg_covar=reg, max_iter=300, random_state=SEED).fit(X)


def scan_k(X, kmax=8, reg=0.05):
    """BIC, AIC, log-likelihood, dan uji rasio likelihood k vs k-1 (k=1 hanya pembanding)."""
    n, d = X.shape
    rows, prev = [], None
    for k in range(1, kmax + 1):
        g = fit_gmm(X, k, reg)
        ll = g.score(X) * n
        row = dict(k=k, loglik=ll, BIC=g.bic(X), AIC=g.aic(X), iter=g.n_iter_, konvergen=bool(g.converged_),
                   LRT=np.nan, df=np.nan, p_value=np.nan)
        if prev is not None:
            df = 1 + d + d * (d + 1) // 2  # parameter tambahan per komponen (bobot + mean + kovarians penuh)
            row.update(LRT=2 * (ll - prev), df=df, p_value=chi2.sf(2 * (ll - prev), df))
        prev = ll
        rows.append(row)
    return pd.DataFrame(rows)


def ellipse_xy(center, cov, level=0.95):
    vals, vecs = np.linalg.eigh(cov)
    r = np.sqrt(chi2.ppf(level, 2))
    th = np.linspace(0, 2 * np.pi, 120)
    pts = vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * r * np.vstack([np.cos(th), np.sin(th)]))
    return center[0] + pts[0], center[1] + pts[1]


def analyze(w, k, reg=0.05):
    f = build_features(w)
    pre = preprocess(f)
    X = pre["scaled"].values
    scan = scan_k(X, 8, reg)
    gm = fit_gmm(X, k, reg)
    means_orig = pre["scaler"].inverse_transform(gm.means_)
    perm = np.argsort(-means_orig[:, FEATURES.index("Mean_FLPR")])  # klaster A = rata-rata FLPR tertinggi
    names = [f"Klaster {chr(65 + i)}" for i in range(k)]
    gro = means_orig[perm, FEATURES.index("Growth_Rate")]
    rest = list(range(1, k))
    fast = max(rest, key=lambda i: gro[i]) if len(rest) > 1 and gro[rest].max() > 2 else None  # transisi cepat: tumbuh > 2 poin
    tiers = ["high"] + ["trans" if i == fast else "low" for i in rest]
    extra, colors = iter(["#FFA94D", "#38BDF8", "#F472B6", "#FBBF24", "#94A3B8"]), []
    for i, t in enumerate(tiers):
        colors.append({"high": "#E85D9E"}.get(t) or ("#34D399" if t == "trans" else "#A78BFA" if tiers[:i].count("low") == 0 else next(extra)))
    gamma = gm.predict_proba(X)[:, perm]
    srt = np.sort(gamma, axis=1)
    pca = PCA(2, random_state=SEED).fit(X)
    Z = pca.transform(X)
    covs = gm.covariances_[perm]
    ell = [ellipse_xy(pca.transform(gm.means_[perm][[i]])[0], pca.components_ @ covs[i] @ pca.components_.T) for i in range(k)]
    labels = gamma.argmax(1)

    res = w.loc[f.index, ["Country Name", "Region", "IncomeGroup"]].copy()
    for i, n in enumerate(names):
        res[n] = gamma[:, i]
    res["Klaster"] = [names[i] for i in labels]
    res["gamma_maks"], res["gamma_kedua"] = srt[:, -1], srt[:, -2]
    res["Borderline"] = (res["gamma_maks"] > BORDER_LO) & (res["gamma_maks"] < BORDER_HI)
    res["PC1"], res["PC2"] = Z[:, 0], Z[:, 1]
    res["Rekomendasi"] = [PROFILES[tiers[i]][2] for i in labels]
    res = res.join(pre["clean"])

    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    ev = dict(loglik=gm.score(X) * len(X), BIC=gm.bic(X), AIC=gm.aic(X), iter=gm.n_iter_, konvergen=bool(gm.converged_),
              silhouette=silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan,
              davies_bouldin=davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan,
              ari_vs_kmeans=adjusted_rand_score(km.labels_, labels))
    return dict(feats=f, pre=pre, scan=scan, gm=gm, k=k, names=names, tiers=tiers, colors=colors, res=res, ev=ev,
                weights=gm.weights_[perm], means=pd.DataFrame(means_orig[perm], index=names, columns=FEATURES),
                covs=covs, ell=ell, pca_var=pca.explained_variance_ratio_, best_k=int(scan[scan.k >= 2].sort_values("BIC").k.iloc[0]))


if __name__ == "__main__":
    for nama, w in [("sintetis", synthetic())] + ([("csv", load_local(os.path.join(os.path.dirname(__file__), "data")))]
                                                   if os.path.isdir(os.path.join(os.path.dirname(__file__), "data")) else []):
        r = analyze(w, 3)
        assert np.allclose(r["res"][r["names"]].sum(axis=1), 1) and r["res"].shape[0] > 50
        assert abs(r["pre"]["scaled"].mean().abs().max()) < 1e-8
        print(nama, r["res"].shape[0], "negara | k-BIC:", r["best_k"], "| borderline:", int(r["res"].Borderline.sum()),
              "|", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["ev"].items()})
        print(r["res"].Klaster.value_counts().to_dict(), r["means"][["Mean_FLPR", "Growth_Rate"]].round(1).to_dict("index"))
    print("OK")