import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="GMM · Partisipasi Kerja Perempuan", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
PINK, GREEN, VIOLET, PEACH = "#E85D9E", "#34D399", "#A78BFA", "#FFA94D"  # tema SweetSoul
PAGES = ["Overview & Data", "Preprocessing & EM Optimization", "Soft Clustering & Ellipses", "Policy Decision Matrix"]
PAGE_SUB = dict(zip(PAGES, ["gambaran data dan pola partisipasi kerja perempuan", "imputasi, inisialisasi EM, dan pemilihan k",
                            "keanggotaan probabilistik dan elips kovarians", "rekomendasi kebijakan per klaster"]))
GRAD_PINK, GRAD_VIOLET = "linear-gradient(90deg,#F8B4D0,#E85D9E)", "linear-gradient(90deg,#D8CCFE,#8B5CF6)"

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,500&display=swap');
:root{--ink:#5B2A45;--pink:#E85D9E}
.stApp{background:linear-gradient(180deg,#FFF7FB 0%,#FDE8F1 100%);font-family:'Nunito',sans-serif}
.stApp p,.stApp label,.stApp li,.stApp button,.stApp input,.stApp textarea,.stApp caption,.stApp small{font-family:'Nunito',sans-serif;color:var(--ink)}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Playfair Display',serif;color:var(--ink);font-weight:700}
[data-testid="stHeader"]{background:transparent}.block-container{padding-top:1.6rem;max-width:1500px}
/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#F9B4CF 0%,#F38CB6 50%,#EA5E99 100%);border:0}
.brand{display:flex;gap:.7rem;align-items:center;margin:.3rem 0 1.2rem}
.brand .logo{width:42px;height:42px;border-radius:50%;background:#fff;color:var(--pink);display:grid;place-items:center;font-size:1.3rem;box-shadow:0 4px 12px rgba(150,40,90,.25)}
.brand b{font-family:'Playfair Display',serif;font-size:1.25rem;color:var(--ink);display:block;line-height:1.1}
.brand small{color:#7A3558;font-size:.72rem}
.sec{font-size:.68rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:#fff;opacity:.9;margin:1rem 0 .4rem .3rem;text-shadow:0 1px 2px rgba(150,40,90,.35)}
.st-key-nav div[role="radiogroup"]{gap:.3rem}
.st-key-nav div[role="radiogroup"] label{padding:.6rem 1rem;border-radius:14px;width:100%;background:transparent;transition:.15s}
.st-key-nav div[role="radiogroup"] label>div:first-child{display:none}
.st-key-nav label p{color:#fff!important;font-weight:700;text-shadow:0 1px 2px rgba(150,40,90,.35)}
.st-key-nav label:has(input:checked){background:#fff;box-shadow:0 4px 14px rgba(150,40,90,.22)}
.st-key-nav label:has(input:checked) p{color:var(--pink)!important;text-shadow:none}
.st-key-ctl{background:rgba(255,255,255,.93);border-radius:18px;padding:.9rem 1rem;box-shadow:0 6px 18px rgba(150,40,90,.18)}
.st-key-ctl label p,.st-key-ctl small{color:var(--ink)!important}
/* header halaman */
.ph{margin:0 0 1rem}.ph h1{font-size:2.1rem;margin:0;line-height:1.1}
.ph span{font-family:'Playfair Display',serif;font-style:italic;color:#B0678C}
/* kartu */
.card,.kpi,.panel{background:#fff;border:1px solid #F6D3E2;border-radius:22px;box-shadow:0 6px 24px rgba(232,93,158,.10);padding:1.1rem 1.3rem;margin-bottom:1rem}
.hero{display:flex;justify-content:space-between;align-items:center;gap:1.5rem;background:linear-gradient(100deg,#EE6A9C 0%,#F48FB1 55%,#F9C0D6 100%);border-radius:26px;padding:1.6rem 2rem;margin-bottom:1.1rem;box-shadow:0 12px 30px rgba(232,93,158,.28)}
.stApp .hero h2{color:#fff;font-size:1.9rem;margin:0 0 .35rem}.stApp .hero p{color:#fff;margin:.2rem 0;max-width:46rem}
.chip{display:inline-block;background:rgba(255,255,255,.92);color:var(--pink);border-radius:999px;padding:.22rem .8rem;font-weight:800;font-size:.76rem;margin:.55rem .4rem 0 0}
.kbox{flex:none;width:104px;height:104px;border-radius:28px;background:rgba(255,255,255,.92);border:4px solid rgba(255,255,255,.55);display:grid;place-items:center;font-family:'Playfair Display',serif;font-weight:700;font-size:2rem;color:var(--pink)}
.kpi{text-align:center;padding:1rem .6rem .8rem;margin-bottom:1.1rem}
.ring{--p:50;--c:#E85D9E;width:78px;height:78px;border-radius:50%;margin:0 auto .5rem;display:grid;place-items:center;position:relative;background:conic-gradient(var(--c) calc(var(--p)*1%),#FBE3EE 0)}
.ring::before{content:"";position:absolute;inset:8px;border-radius:50%;background:#fff}
.ring span{position:relative;font-weight:800;font-size:.9rem;color:var(--ink)}
.kpi small{display:block;font-weight:700;color:#A8607F;font-size:.8rem}
/* panel kanan */
.pt{font-family:'Playfair Display',serif;font-weight:700;font-size:1.1rem;margin:0 0 .15rem}
.hint{font-size:.74rem;color:#B0678C;font-style:italic;margin-bottom:.5rem}
.mrow{display:grid;grid-template-columns:minmax(70px,40%) 1fr 38px;gap:8px;align-items:center;font-size:.8rem;margin:.4rem 0}
.mrow span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.mrow b{text-align:right;color:var(--pink)}
.bar{height:10px;border-radius:99px;background:#FBE3EE}.bar>i{display:block;height:100%;border-radius:99px}
.pill{background:#FDE8F1;color:var(--pink);border-radius:99px;padding:.05rem .55rem;font-weight:800;font-size:.76rem}
.brow{display:flex;justify-content:space-between;align-items:center;padding:.3rem 0;border-bottom:1px dashed #F3C9DB;font-size:.84rem}
.nudge{background:linear-gradient(135deg,#F9C9DE 0%,#E3D7FD 100%);border:1px solid #F6D3E2;border-radius:22px;padding:1.1rem 1.3rem;margin-bottom:1rem;box-shadow:0 6px 24px rgba(232,93,158,.10)}
.gentle{display:flex;align-items:center;gap:.9rem;margin:1.4rem 0 .9rem;font-family:'Playfair Display',serif;font-style:italic;color:#C0709A}
.gentle::before,.gentle::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,transparent,#E9A9C6,transparent)}
.stButton>button,.stDownloadButton>button{background:linear-gradient(180deg,#F8A5C8,#E85D9E);border:0;border-radius:999px;font-weight:800}
.stButton>button p,.stDownloadButton>button p{color:#fff!important}
</style>""", unsafe_allow_html=True)


def style(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFAFC",
                      font=dict(family="Nunito, sans-serif", color="#5B2A45"), margin=dict(l=10, r=10, t=50, b=10),
                      colorway=[PINK, VIOLET, GREEN, PEACH])
    return fig


def card(html, cls="card"):
    st.markdown(f"<div class='{cls}'>{html}</div>", unsafe_allow_html=True)


def bars(series, grad):
    p = series / series.sum()
    return "".join(f"<div class='mrow'><span title='{i}'>{i}</span><div class='bar'><i style='width:{v*100:.0f}%;background:{grad}'></i></div><b>{v:.0%}</b></div>"
                   for i, v in p.items())


def gentle(text):
    st.markdown(f"<div class='gentle'>{text}</div>", unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='brand'><div class='logo'>&#9792;</div><div><b>FLPR Atlas</b><small>soft clustering studio</small></div></div>"
                "<div class='sec'>Analisis</div>", unsafe_allow_html=True)
    with st.container(key="nav"):
        page = st.radio("Navigasi", PAGES, label_visibility="collapsed")
    st.markdown("<div class='sec'>Model &amp; data</div>", unsafe_allow_html=True)
    with st.container(key="ctl"):
        k = st.select_slider("Jumlah klaster (k)", options=list(range(2, 9)), value=3)
        reg = st.select_slider("Regularisasi kovarians (reg_covar)", options=[1e-6, 1e-3, 1e-2, 5e-2, 1e-1], value=5e-2,
                               help="Fitur level FLPR berkorelasi ~0,98; nilai kecil membuat elips degenerat dan klaster tidak bermakna.")
        mode = st.radio("Sumber data", ["Otomatis", "Unggah CSV", "Data sintetis"])
        up_api = up_meta = None
        if mode == "Unggah CSV":
            up_api = st.file_uploader("CSV data indikator (API_...csv)", type="csv")
            up_meta = st.file_uploader("CSV metadata negara (Metadata_Country_...csv)", type="csv")


@st.cache_data(show_spinner="Memuat data...")
def get_data(mode, a, m):
    if mode == "Data sintetis":
        return core.synthetic(), "Data sintetis"
    if mode == "Unggah CSV":
        return core.load_csv(io.BytesIO(a), io.BytesIO(m)), "CSV unggahan"
    try:
        return core.load_api(), "API World Bank"
    except Exception:
        try:
            return core.load_local(os.path.join(HERE, "data")), "CSV lokal (API tidak terjangkau)"
        except Exception:
            return core.synthetic(), "Data sintetis (API & CSV tidak tersedia)"


@st.cache_data(show_spinner="Melatih GMM (EM)...")
def run(w, k, reg):
    return core.analyze(w, k, reg)


if mode == "Unggah CSV" and not (up_api and up_meta):
    st.info("Unggah kedua berkas CSV di sidebar untuk melanjutkan.")
    st.stop()
wide, source = get_data(mode, up_api.getvalue() if up_api else None, up_meta.getvalue() if up_meta else None)
R = run(wide, k, reg)
res, ev, names, colors, tiers = R["res"], R["ev"], R["names"], R["colors"], R["tiers"]
cmap = dict(zip(names, colors))
nb = int(res.Borderline.sum())
pname = lambda n: core.PROFILES[tiers[names.index(n)]][0]


def right_panel():
    b = res[res.Borderline].sort_values("gamma_maks").head(8)
    rows = "".join(f"<div class='brow'><span>{i['Country Name']}</span><span class='pill'>{i['gamma_maks']:.2f} / {i['gamma_kedua']:.2f}</span></div>"
                   for _, i in b.iterrows()) or "Tidak ada negara borderline."
    card(f"<div class='pt'>Negara borderline ({nb})</div><div class='hint'>γ tertinggi / kedua · ketidakpastian aleatorik</div>{rows}", "panel")
    card("<div class='pt'>Distribusi klaster</div><div class='hint'>proporsi negara per klaster</div>" + bars(res.Klaster.value_counts().sort_index(), GRAD_PINK) +
         "<div class='pt' style='margin-top:1rem'>Distribusi IncomeGroup</div><div class='hint'>proporsi negara per kelompok pendapatan</div>" + bars(res.IncomeGroup.value_counts(), GRAD_VIOLET), "panel")
    low = names[tiers.index("low")]
    card(f"<div class='pt'>Sorotan kebijakan · {low}</div><div class='hint'>{pname(low)}</div>{core.PROFILES['low'][2]}", "nudge")


def kpis():
    sc = R["scan"][R["scan"].k >= 2]
    span = sc.BIC.max() - sc.BIC.min()
    items = [(f"{ev['BIC']:,.0f}", "BIC · k aktif", 100 * (sc.BIC.max() - ev["BIC"]) / span if span else 100, PINK),
             (f"{ev['silhouette']:.2f}", "silhouette", max(0, ev["silhouette"]) * 100, VIOLET),
             (f"{len(res)}", "negara dianalisis", 100 * len(res) / len(wide), GREEN),
             (f"{nb}", "negara borderline", 100 * nb / len(res), "#38BDF8")]
    for col, (v, lab, pct, c) in zip(st.columns(4), items):
        col.markdown(f"<div class='kpi'><div class='ring' style='--p:{pct:.0f};--c:{c}'><span>{v}</span></div><small>{lab}</small></div>", unsafe_allow_html=True)


main, side = st.columns([3.2, 1.2])
with main:
    st.markdown(f"<div class='ph'><h1>{page}</h1><span>{PAGE_SUB[page]}</span></div>", unsafe_allow_html=True)
    top = res.sort_values("gamma_maks").iloc[0]
    chips = "".join(f"<span class='chip'>{c}</span>" for c in ["covariance full", "init K-Means", f"logL {ev['loglik']:,.1f}",
                    f"{'konvergen' if ev['konvergen'] else 'belum konvergen'} · {ev['iter']} iterasi", source])
    card(f"<div><h2>Probabilistic Soft Clustering</h2><p>{len(res)} negara terbagi ke {k} klaster; {nb} negara borderline "
         f"(0,35 &lt; γ &lt; 0,65), paling ambigu: {top['Country Name']}. BIC terendah pada k={R['best_k']}.</p>{chips}</div>"
         f"<div class='kbox'>k={k}</div>", "hero")
    kpis()

    # ===== Halaman 1: tahap 1 =====
    if page == PAGES[0]:
        st.subheader("Tahap 1 · Eksplorasi karakteristik dataset")
        yrs = list(range(1990, 2024))
        ts = wide[yrs]
        q = ts.quantile([.1, .5, .9]).T
        fig = go.Figure()
        fig.add_scatter(x=yrs, y=q[.9], line=dict(width=0), showlegend=False)
        fig.add_scatter(x=yrs, y=q[.1], fill="tonexty", fillcolor="rgba(232,93,158,.15)", line=dict(width=0), name="Persentil 10–90")
        fig.add_scatter(x=yrs, y=q[.5], line=dict(color=PINK, width=3), name="Median dunia")
        for reg_name, g in wide.groupby("Region"):
            fig.add_scatter(x=yrs, y=g[yrs].median(), name=reg_name, line=dict(width=1.3), visible="legendonly")
        st.plotly_chart(style(fig.update_layout(title="Distribusi temporal FLPR (% populasi perempuan 15+)", yaxis_title="FLPR (%)")), use_container_width=True)
        c1, c2 = st.columns(2)
        yr = c1.select_slider("Tahun histogram", options=yrs, value=2023)
        c1.plotly_chart(style(px.histogram(wide, x=yr, nbins=30, color_discrete_sequence=[PINK], title=f"Sebaran FLPR {yr}"), 320), use_container_width=True)
        c2.plotly_chart(style(px.box(wide.reset_index(), x="Region", y=yr, color="Region", title=f"FLPR {yr} per Region").update_layout(showlegend=False, xaxis_title=None), 320), use_container_width=True)
        st.markdown("**Statistik deskriptif fitur turunan (sebelum imputasi)**")
        st.dataframe(R["feats"].describe().T.round(2), use_container_width=True)
        st.caption(f"Cakupan: {len(wide)} negara (agregat wilayah dikeluarkan); {len(R['feats'])} negara punya data cukup untuk pemodelan. "
                   "Mean_FLPR dan Std_Dev_FLPR memakai 2014–2023; Growth_Rate = FLPR_2023 − FLPR_2013.")

    # ===== Halaman 2: tahap 2-5 =====
    elif page == PAGES[1]:
        pre = R["pre"]
        st.subheader("Tahap 2 · Preprocessing & imputasi")
        miss = pd.DataFrame({"Sebelum": pre["raw"].isna().sum(), "Sesudah": pre["imputed"].isna().sum()})
        c1, c2 = st.columns([1, 2])
        c1.dataframe(miss, use_container_width=True)
        feat = c2.selectbox("Bandingkan fitur", core.FEATURES, index=4)
        cmp = pd.concat([pd.DataFrame({"nilai": pre["raw"][feat], "tahap": "Asli (ada NaN)"}),
                         pd.DataFrame({"nilai": pre["clean"][feat], "tahap": "Imputasi + winsorisasi"}),
                         pd.DataFrame({"nilai": pre["scaled"][feat], "tahap": "Z-score"})])
        c2.plotly_chart(style(px.box(cmp, x="tahap", y="nilai", color="tahap", color_discrete_sequence=[GREEN, VIOLET, PINK]).update_layout(showlegend=False), 300), use_container_width=True)
        st.caption(f"Imputasi: IterativeImputer (BayesianRidge) menjaga korelasi antarfitur. Pencilan ekstrem di-winsorisasi pada Q1−3·IQR / Q3+3·IQR ({pre['n_clipped']} nilai). "
                   "Rata-rata Z-score ≈ 0 dan simpangan baku = 1 per fitur.")
        st.dataframe(pre["scaled"].describe().loc[["mean", "std"]].round(3), use_container_width=True)

        st.subheader("Tahap 3 · Inisialisasi EM")
        st.markdown("Model dilatih dengan `init_params='kmeans'`: pusat klaster K-Means menjadi **μ_old** awal, "
                    "dengan 10 restart (`n_init=10`) untuk mengurangi risiko konvergensi ke local optimum.")

        st.subheader("Tahap 4 · Pemodelan EM & penentuan k optimal")
        sc = R["scan"][R["scan"].k >= 2]
        fig = go.Figure()
        fig.add_scatter(x=sc.k, y=sc.BIC, name="BIC", mode="lines+markers", line=dict(color=PINK, width=3))
        fig.add_scatter(x=sc.k, y=sc.AIC, name="AIC", mode="lines+markers", line=dict(color=PEACH, width=2))
        fig.add_scatter(x=sc.k, y=sc.loglik, name="Log-Likelihood", mode="lines+markers", line=dict(color=VIOLET, width=3), yaxis="y2")
        fig.update_layout(title="BIC, AIC, dan Log-Likelihood terhadap k", xaxis_title="k", yaxis_title="BIC / AIC",
                          yaxis2=dict(title="Log-Likelihood", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(style(fig), use_container_width=True)
        t = R["scan"].set_index("k").round(3)
        t["LRT (−2 log λ vs k−1)"], t["Signifikan (p<0,05)"] = t.pop("LRT"), t["p_value"] < 0.05
        st.dataframe(t, use_container_width=True)
        st.caption(f"BIC minimum pada k={R['best_k']}; analisis aktif memakai k={k} (default studi kasus k=3). "
                   "p-value LRT bersifat indikatif karena syarat regularitas χ² tidak terpenuhi pada uji jumlah komponen mixture "
                   "(McLachlan & Rathnayake, 2014); keputusan akhir ada pada peneliti.")

        st.subheader("Tahap 5 · Evaluasi model")
        c = st.columns(4)
        for col, (a, b) in zip(c, [("Log-Likelihood", f"{ev['loglik']:,.1f}"), ("Silhouette", f"{ev['silhouette']:.3f}"),
                                   ("Davies-Bouldin", f"{ev['davies_bouldin']:.3f}"), ("ARI vs K-Means", f"{ev['ari_vs_kmeans']:.3f}")]):
            col.markdown(f"<div class='kpi'><small>{a}</small><div>{b}</div></div>", unsafe_allow_html=True)
        st.caption("Silhouette tinggi dan Davies-Bouldin rendah menandakan klaster lebih terpisah. Silhouette dihitung dari label keras (argmax γ), "
                   "sehingga tidak menangkap ketidakpastian soft clustering; ARI rendah berarti GMM full-covariance berbeda nyata dari partisi bola K-Means.")

    # ===== Halaman 3: tahap 6 =====
    elif page == PAGES[2]:
        st.subheader("Tahap 6 · Interpretasi klaster & soft clustering")
        fig = go.Figure()
        alpha = 0.25 + 0.75 * res.gamma_maks
        for n in names:
            d = res[res.Klaster == n]
            rgb = tuple(int(cmap[n][i:i + 2], 16) for i in (1, 3, 5))
            fig.add_scatter(x=d.PC1, y=d.PC2, mode="markers", name=f"{n} · {pname(n)}", text=d["Country Name"],
                            marker=dict(size=9, color=[f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.2f})" for a in alpha[d.index]], line=dict(width=0.5, color="white")),
                            customdata=d[names].values,
                            hovertemplate="<b>%{text}</b><br>" + "<br>".join(f"γ {n2}: %{{customdata[{i}]:.2f}}" for i, n2 in enumerate(names)) + "<extra></extra>")
        for n, (ex, ey) in zip(names, R["ell"]):
            fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=cmap[n], width=2.5, dash="dash"), showlegend=False, hoverinfo="skip")
        b = res[res.Borderline]
        fig.add_scatter(x=b.PC1, y=b.PC2, mode="markers", name="Borderline", text=b["Country Name"], hoverinfo="text",
                        marker=dict(size=15, symbol="circle-open", color="#5B2A45", line=dict(width=2)))
        v = R["pca_var"]
        st.plotly_chart(style(fig.update_layout(title="Proyeksi PCA 2D dengan elips kovarians 95% (Σ_k); transparansi titik = γ maks",
                                                xaxis_title=f"PC1 ({v[0]:.0%})", yaxis_title=f"PC2 ({v[1]:.0%})"), 560), use_container_width=True)
        st.markdown("**Vektor rata-rata μ_k (satuan asli, % / poin) dan bobot campuran π_k**")
        mt = R["means"].round(2)
        mt.insert(0, "π_k", R["weights"].round(3))
        mt.insert(0, "Profil", [pname(n) for n in names])
        st.dataframe(mt, use_container_width=True)
        c1, c2 = st.columns(2)
        sel = c1.selectbox("Matriks kovarians Σ_k (ruang Z-score) untuk", names)
        i = names.index(sel)
        c1.plotly_chart(style(px.imshow(pd.DataFrame(R["covs"][i], index=core.FEATURES, columns=core.FEATURES).round(2), text_auto=True,
                                        color_continuous_scale=[[0, "#FFF0F6"], [.5, "#F7A1C4"], [1, "#E85D9E"]]), 380), use_container_width=True)
        c2.plotly_chart(style(px.histogram(res, x="gamma_maks", nbins=20, color_discrete_sequence=[PINK],
                                           title="Sebaran γ maksimum (responsibility)").add_vrect(x0=.35, x1=.65, fillcolor=GREEN, opacity=.3, line_width=0), 380), use_container_width=True)
        gentle(f"Negara borderline · {nb} negara (0,35 &lt; γ maks &lt; 0,65)")
        st.dataframe(b.sort_values("gamma_maks")[["Country Name", "Region"] + names + ["Klaster"]].round(3), use_container_width=True)
        st.caption("K-Means memberi label tegas tanpa derajat keyakinan; GMM menunjukkan negara yang keanggotaannya terbagi, "
                   "sehingga kebijakan untuk negara tersebut sebaiknya memadukan profil dua klaster teratas.")

    # ===== Halaman 4: tahap 7 =====
    else:
        st.subheader("Tahap 7 · Matriks keputusan & rekomendasi kebijakan")
        rows = []
        for n in names:
            d = res[res.Klaster == n]
            rows.append({"Klaster": n, "Profil": pname(n), "Negara": len(d), "Mean FLPR": d.Mean_FLPR.mean().round(1),
                         "Growth (poin)": d.Growth_Rate.mean().round(1), "Volatilitas": d.Std_Dev_FLPR.mean().round(2),
                         "Region dominan": d.Region.mode().iloc[0], "Income dominan": d.IncomeGroup.mode().iloc[0],
                         "Rekomendasi": core.PROFILES[tiers[names.index(n)]][2]})
        st.dataframe(pd.DataFrame(rows).set_index("Klaster"), use_container_width=True)
        c1, c2 = st.columns(2)
        dim = c1.radio("Dimensi", ["Region", "IncomeGroup"], horizontal=True)
        fig = px.pie(res, names="Klaster", hole=.55, color="Klaster", color_discrete_map=cmap, title="Proporsi negara per klaster")
        c2.plotly_chart(style(fig, 330), use_container_width=True)
        ct = pd.crosstab(res[dim], res.Klaster)
        c1.plotly_chart(style(px.bar(ct.div(ct.sum(axis=1), axis=0).reset_index().melt(id_vars=dim), y=dim, x="value", color="Klaster", orientation="h",
                                     color_discrete_map=cmap, title=f"Komposisi klaster per {dim}").update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None), 330), use_container_width=True)
        n_sel = st.selectbox("Donut komposisi dimensi di dalam klaster", names)
        sub = res[res.Klaster == n_sel][dim].value_counts().reset_index()
        st.plotly_chart(style(px.pie(sub, names=dim, values="count", hole=.55, color_discrete_sequence=[PINK, GREEN, VIOLET, PEACH, "#38BDF8", "#F472B6", "#FBBF24"],
                                     title=f"{n_sel} · {pname(n_sel)}: sebaran {dim}"), 330), use_container_width=True)
        gentle("Tabel probabilitas posterior γ(z_nk) per negara")
        f1, f2 = st.columns(2)
        regs = f1.multiselect("Filter Region", sorted(res.Region.unique()))
        only_b = f2.checkbox("Hanya negara borderline")
        tab = res[(res.Region.isin(regs) if regs else pd.Series(True, index=res.index)) & (res.Borderline if only_b else pd.Series(True, index=res.index))]
        show = tab[["Country Name", "Region", "IncomeGroup"] + names + ["Klaster", "Rekomendasi"]]
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={n: st.column_config.ProgressColumn(f"γ {n}", min_value=0, max_value=1, format="%.2f") for n in names})
        st.download_button("Unduh tabel (CSV)", show.to_csv(index=False).encode("utf-8"), "gamma_negara.csv", "text/csv")
with side:
    right_panel()