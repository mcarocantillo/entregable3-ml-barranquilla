"""Convierte los cuadernos (.py en formato jupytext), los ejecuta y construye el libro.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Uso (dentro de la carpeta libro/):  python construir_libro.py   (Entregable 3)
Requiere: jupytext, nbclient, ipykernel y jupyter-book 1.x, además de las
dependencias del dashboard (requirements.txt de la carpeta superior).
"""
import subprocess
import sys
from pathlib import Path

import jupytext
from nbclient import NotebookClient

AQUI = Path(__file__).resolve().parent
for py in sorted(p for p in (AQUI / "notebooks").glob("*.py") if not p.name.startswith("_")):
    nb = jupytext.read(py)
    print(f"Ejecutando {py.name} ...", flush=True)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(py.parent)}}).execute()
    jupytext.write(nb, py.with_suffix(".ipynb"))
subprocess.run([sys.executable, "-c", "from jupyter_book.cli.main import main; main()", "build", str(AQUI)],
               check=True)
print("Libro en", AQUI / "_build" / "html" / "index.html")
