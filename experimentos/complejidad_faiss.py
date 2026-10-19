# -*- coding: utf-8 -*-
"""KNN exacto frente a vecinos aproximados con FAISS (sección 4.2 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Compara, con las mismas p variables preprocesadas del pipeline y n de 4 000 a 256 000 viviendas:

- scikit-learn, fuerza bruta (exacto, O(n·p) por consulta) y KD-Tree;
- FAISS ``IndexFlatL2`` (exacto, fuerza bruta con BLAS);
- FAISS ``IndexIVFFlat`` (aproximado: agrupa con k-medias en √n listas y solo busca en las ``nprobe`` más cercanas;
  costo ≈ O((n/√n)·nprobe·p) por consulta);
- FAISS ``IndexHNSWFlat`` (aproximado: grafo jerárquico de mundo pequeño; costo ≈ O(log n) por consulta).

Para cada uno: tiempo de construcción, tiempo de búsqueda de 2 000 consultas (k = 30), tamaño del índice,
**recall@30** (fracción de los 30 vecinos exactos que el método encuentra) y el F1 macro del voto de los 30 vecinos.
Todo en un hilo (``faiss.omp_set_num_threads(1)``) para comparar algoritmos, no paralelismo.

Salida: ``resultados/complejidad_faiss.json`` y ``resultados/figuras/faiss_knn.png``.
Uso (desde e3/, con el computador libre):  ../venv_e2/Scripts/python -u experimentos/complejidad_faiss.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402

warnings.filterwarnings("ignore")
FIG = CF.RESULTADOS / "figuras"
TAMANOS = [4000, 16000, 64000, 256000]
K = 30


def voto(vecinos, ytr, clases=CF.CLASES):
    """Estrato más frecuente entre los vecinos de cada consulta."""
    cuentas = np.stack([(ytr[vecinos] == k).sum(1) for k in clases], 1)
    return clases[cuentas.argmax(1)]


def main():
    import argparse
    import faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanos", nargs="*", type=int, default=TAMANOS)
    ap.add_argument("--salida", default="complejidad_faiss.json")
    args = ap.parse_args()
    tamanos = args.tamanos
    faiss.omp_set_num_threads(1)
    from estrato3 import datos as DT
    CF.fijar_semillas()
    d = DT.cargar()
    tr = d["train"].sample(frac=1, random_state=CF.SEED).reset_index(drop=True)
    prep = MD.preprocesador(d).fit(tr.iloc[:32000])
    X = np.ascontiguousarray(prep.transform(tr.iloc[:max(tamanos)]), dtype=np.float32)
    y = tr.estrato_num.to_numpy()[:max(tamanos)]
    te = d["test"].sample(2000, random_state=CF.SEED)
    Q = np.ascontiguousarray(prep.transform(te), dtype=np.float32)
    yq = te.estrato_num.to_numpy()
    p = X.shape[1]
    print(f"p = {p}; consultas = {len(Q)}; k = {K}", flush=True)
    filas = []
    for n in tamanos:
        Xn, yn = X[:n], y[:n]
        exactos = None
        metodos = {
            "scikit-learn fuerza bruta": None, "scikit-learn KD-Tree": None, "FAISS Flat (exacto)": None,
            f"FAISS IVF (nlist = {int(np.sqrt(n))}, nprobe = 8)": None, "FAISS HNSW (M = 32, efSearch = 64)": None}
        for nombre in metodos:
            t0 = time.perf_counter()
            if nombre.startswith("scikit"):
                alg = "brute" if "bruta" in nombre else "kd_tree"
                nn = NearestNeighbors(n_neighbors=K, algorithm=alg, n_jobs=1).fit(Xn)
                t_build = time.perf_counter() - t0
                t1 = time.perf_counter()
                I = nn.kneighbors(Q, return_distance=False)
                t_search = time.perf_counter() - t1
                tam = Xn.nbytes / 1e6
            else:
                if "Flat" in nombre:
                    idx = faiss.IndexFlatL2(p)
                elif "IVF" in nombre:
                    idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(p), p, int(np.sqrt(n)))
                    idx.train(Xn)
                    idx.nprobe = 8
                else:
                    idx = faiss.IndexHNSWFlat(p, 32)
                    idx.hnsw.efSearch = 64
                idx.add(Xn)
                t_build = time.perf_counter() - t0
                t1 = time.perf_counter()
                _, I = idx.search(Q, K)
                t_search = time.perf_counter() - t1
                tam = len(faiss.serialize_index(idx)) / 1e6
            if nombre == "scikit-learn fuerza bruta":
                exactos = I
            recall = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(I, exactos)]))
            f1 = f1_score(yq, voto(I, yn), average="macro", zero_division=0)
            familia = nombre.split(" (")[0]
            filas.append({"metodo": familia, "detalle": nombre, "n": n, "seg_construccion": t_build,
                          "seg_busqueda_2000": t_search, "ms_por_consulta": 1000 * t_search / len(Q),
                          "indice_MB": tam, "recall_at_30": recall, "f1_macro_voto": f1})
            print(f"n={n:7d} {nombre:42s} construir {t_build:7.2f} s | buscar {t_search:7.2f} s | "
                  f"{tam:7.1f} MB | recall {recall:.3f} | F1 {f1:.3f}", flush=True)
    exps = {}
    for fam in dict.fromkeys(f["metodo"] for f in filas):
        sel = [f for f in filas if f["metodo"] == fam]
        ns = np.log([f["n"] for f in sel])
        exps[fam] = float(np.polyfit(ns, np.log([f["seg_busqueda_2000"] for f in sel]), 1)[0])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for fam in exps:
        sel = [f for f in filas if f["metodo"] == fam]
        ax[0].loglog([f["n"] for f in sel], [f["ms_por_consulta"] for f in sel], marker="o",
                     label=f"{fam} (b = {exps[fam]:.2f})")
        ax[1].plot([f["n"] for f in sel], [f["recall_at_30"] for f in sel], marker="o", label=fam)
    ax[0].set_xlabel("viviendas indexadas (n)")
    ax[0].set_ylabel("ms por consulta (escala log)")
    ax[0].set_title("Costo de búsqueda: exacto frente a aproximado")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("viviendas indexadas (n)")
    ax[1].set_ylabel("recall@30 frente a la búsqueda exacta")
    ax[1].set_title("Exactitud de los vecinos aproximados")
    for a in ax:
        a.legend(fontsize=8)
    plt.savefig(FIG / ("faiss_knn.png" if args.salida == "complejidad_faiss.json" else "faiss_prueba.png"), dpi=130, bbox_inches="tight")
    plt.close()
    (CF.RESULTADOS / args.salida).write_text(
        json.dumps({"p": p, "k": K, "consultas": len(Q), "mediciones": filas, "exponentes_busqueda": exps,
                    "faiss_version": faiss.__version__}, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print("guardado complejidad_faiss.json", flush=True)


if __name__ == "__main__":
    main()
