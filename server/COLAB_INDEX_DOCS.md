# Ejecutar `index_docs.py` en Google Colab

Instrucciones rápidas para ejecutar el indexador de three.js en Colab sin cargar tu máquina.

1) Abrir Google Colab: https://colab.research.google.com/
2) Crear un nuevo notebook y pegar las siguientes celdas (ejecutar en orden).

---

```bash
# 1) Actualizar pip
!python -m pip install --upgrade pip

# 2) Instalar dependencias (usa el requirements del repo)
!pip install -r https://raw.githubusercontent.com/Hardwaretor/AI/refs/heads/main/server/requirements.txt

# 3) Clonar repo (opcional si subes manualmente los archivos)
!git clone https://github.com/Hardwaretor/AI.git repo || true
%cd repo/server

# 4) Ejecutar indexador en modo seguro (sin FAISS, solo descarga y embeddings en lotes)
!python index_docs.py --output-dir ./data --max-pages 200 --no-faiss --download-only --delay 0.5

# 5) Después de ejecutar, puedes descargar el contenido de ./data usando la UI de Colab
```

Notas:
- Ajusta `--max-pages` a un número menor (10, 50) si quieres probar primero.
- `--download-only` evitará la generación de embeddings/FAISS; sube a `--no-faiss` o quita la flag para generar embeddings si tu entorno tiene RAM suficiente.
- Si `faiss-cpu` falla en Colab, puedes ejecutar sin FAISS y guardar embeddings en numpy.

---

¿Quieres que genere un notebook `.ipynb` en el repo también? Si sí, lo creo aquí mismo.
