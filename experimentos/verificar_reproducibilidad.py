# -*- coding: utf-8 -*-
"""Verificación de reproducibilidad: repite la búsqueda final de dos finalistas en procesos nuevos.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Repite ``experimento.finalista`` para Random Forest y XGBoost (los dos con aleatoriedad en el modelo, el
balanceo y la búsqueda) y compara hiperparámetros y F1 de test con ``finalistas.json``. Si coinciden, la semilla
única fija todo el resultado; si no, se reporta la diferencia.

Salida: ``resultados/reproducibilidad.json``.
Uso (desde e3/):  ../venv_e2/Scripts/python -u experimentos/verificar_reproducibilidad.py
"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import evaluacion as EV  # noqa: E402
from estrato3 import experimento as EX  # noqa: E402


def main():
    from joblib import Parallel, delayed
    FN = json.loads((CF.RESULTADOS / "finalistas.json").read_text(encoding="utf-8"))["finalistas"]
    modelos = ["Random Forest", "XGBoost"]
    res = Parallel(n_jobs=2, backend="loky")(delayed(EX.finalista)(m, FN[m]["balanceo"], FN[m]["metodo"])
                                             for m in modelos)
    yte = EX._test().estrato_num.to_numpy()
    salida = {}
    for r in res:
        m = r["modelo"]
        f1 = EV.metricas(yte, r["proba_test"])["f1_macro"]
        salida[m] = {"params_antes": FN[m]["params"], "params_ahora": r["params"],
                     "mismos_params": r["params"] == FN[m]["params"], "f1_test_antes": FN[m]["test"]["f1_macro"],
                     "f1_test_ahora": f1, "diferencia": f1 - FN[m]["test"]["f1_macro"]}
        print(f"{m}: params iguales = {salida[m]['mismos_params']}; F1 test {FN[m]['test']['f1_macro']:.4f} → "
              f"{f1:.4f}", flush=True)
    (CF.RESULTADOS / "reproducibilidad.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1,
                                                                    default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
