import os, sys, json, warnings
import numpy as np, pandas as pd, joblib

warnings.filterwarnings("ignore")

# Semua path relatif terhadap folder file ini
BASE = os.path.dirname(os.path.abspath(__file__))

# ===== PENGATURAN (ubah di sini saja) =====
CSV       = os.path.join(BASE, "data_kemiskinan_sumut_2025.csv")
GDRIVE_ID = "14dS7ucQ2txS6se1ulPyjj2aMx0N7yTiM"          # dipakai jika CSV belum ada
TARGET    = "kategori_kemiskinan"
DROP      = ["kabupaten_kota", "persentase_miskin_pct"]   # kolom ID + kolom yang membocorkan label
PILIH     = ["knn", "svm"]                                # pilih 2 dari: knn, nb, tree, svm
REPEATS   = 10                                            # 1 = 5-fold biasa; >1 = diulang
OUT       = os.path.join(BASE, "hasil")
SEED      = 42
# ==========================================

NAMA = {"knn": "KNN", "svm": "SVM (RBF)", "nb": "Naive Bayes", "tree": "Decision Tree"}


# ---------------------------------------------------------------
# 1. LATIH MODEL
# ---------------------------------------------------------------
def latih():
    import matplotlib
    matplotlib.use("Agg")  # simpan gambar tanpa membuka jendela
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                         cross_validate, cross_val_predict)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC

    models = {
        "knn":  ("KNN", lambda: KNeighborsClassifier(n_neighbors=5)),
        "nb":   ("Naive Bayes", lambda: GaussianNB()),
        "tree": ("Decision Tree", lambda: DecisionTreeClassifier(max_depth=6, random_state=SEED)),
        "svm":  ("SVM (RBF)", lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
    }

    def build_pre(X, scale=True):
        num = X.select_dtypes(include="number").columns.tolist()
        cat = [c for c in X.columns if c not in num]
        steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
        pre = ColumnTransformer([
            ("num", Pipeline(steps), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
        return pre, num, cat

    # Unduh dataset otomatis bila belum ada
    if not os.path.exists(CSV):
        import gdown
        print("Mengunduh dataset...")
        gdown.download(id=GDRIVE_ID, output=CSV, quiet=False)

    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV).dropna(subset=[TARGET])
    y = df[TARGET].astype(str)
    X = df.drop(columns=[TARGET] + DROP)
    print("Fitur dipakai:", list(X.columns))
    print("Distribusi kelas:\n", y.value_counts(), "\n")

    cv = (StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) if REPEATS == 1
          else RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=SEED))
    cv_cm = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision_macro",
               "recall": "recall_macro", "f1": "f1_macro"}

    rows, best = [], None
    for key in PILIH:
        name, mk = models[key]
        pre, num, cat = build_pre(X, scale=(key != "tree"))
        pipe = Pipeline([("pre", pre), ("clf", mk())])
        r = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        row = {"Model": name}
        for m in scoring:
            row[f"{m}_mean"] = r[f"test_{m}"].mean()
            row[f"{m}_std"] = r[f"test_{m}"].std()
        rows.append(row)

        pred = cross_val_predict(pipe, X, y, cv=cv_cm)
        labels = sorted(y.unique())
        ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=labels),
                               display_labels=labels).plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name} (5-fold CV)"); plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"cm_{key}.png"), dpi=200); plt.close()
        with open(os.path.join(OUT, f"report_{key}.txt"), "w") as f:
            f.write(classification_report(y, pred, zero_division=0))

        if best is None or row["f1_mean"] > best[0]:
            best = (row["f1_mean"], key, pipe, num, cat)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "perbandingan.csv"), index=False)

    _, key, pipe, num, cat = best
    pipe.fit(X, y)
    joblib.dump(pipe, os.path.join(OUT, "model.joblib"))
    meta = {"model": models[key][0], "target": TARGET, "classes": sorted(y.unique()),
            "numeric": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in num},
            "categorical": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in cat}}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Perbandingan model:\n", res.round(4).to_string(index=False))
    print(f"\nSelesai. Model terbaik (F1 macro): {models[key][0]}")


# ---------------------------------------------------------------
# 2. LIHAT HASIL
# ---------------------------------------------------------------
def lihat_hasil():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    if not os.path.isdir(OUT):
        sys.exit("Folder hasil/ belum ada. Jalankan dulu: python main_utuh.py latih")

    print("Isi folder hasil:", os.listdir(OUT))
    print(pd.read_csv(os.path.join(OUT, "perbandingan.csv")).round(3).to_string(index=False))

    for k in PILIH:
        print(f"\n=== {NAMA[k]} ===")
        print(open(os.path.join(OUT, f"report_{k}.txt")).read())
        plt.figure(figsize=(6, 5))
        plt.imshow(mpimg.imread(os.path.join(OUT, f"cm_{k}.png")))
        plt.axis("off"); plt.title(NAMA[k]); plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 3. DEMO STREAMLIT
# ---------------------------------------------------------------
def app():
    import streamlit as st

    st.set_page_config(page_title="Demo Klasifikasi Sumut")
    if not os.path.exists(os.path.join(OUT, "model.joblib")):
        st.error("Model belum ada. Jalankan dulu: python main_utuh.py latih")
        st.stop()

    meta = json.load(open(os.path.join(OUT, "meta.json")))
    model = joblib.load(os.path.join(OUT, "model.joblib"))

    st.title("Demo Prediksi - Dataset Lokal Sumut")
    st.caption(f"Model: {meta['model']} | Target: {meta['target']}")

    inp = {}
    for c, (lo, hi, med) in meta["numeric"].items():
        inp[c] = st.number_input(c, value=float(med), min_value=float(lo), max_value=float(hi))
    for c, opts in meta["categorical"].items():
        inp[c] = st.selectbox(c, opts)

    if st.button("Prediksi"):
        X = pd.DataFrame([inp])
        pred = model.predict(X)[0]
        st.success(f"Prediksi: **{pred}**")
        if hasattr(model, "predict_proba"):
            st.bar_chart(pd.Series(model.predict_proba(X)[0], index=model.classes_))

    st.info("Catatan etika: model dilatih dari data terbatas; jangan dipakai untuk keputusan yang mendiskriminasi.")


# ---------------------------------------------------------------
# PINTU MASUK
# ---------------------------------------------------------------
def _dalam_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _dalam_streamlit():
    app()
elif __name__ == "__main__":
    perintah = sys.argv[1] if len(sys.argv) > 1 else ""
    if perintah == "latih":
        latih()
    elif perintah == "hasil":
        lihat_hasil()
    else:
        print(__doc__)