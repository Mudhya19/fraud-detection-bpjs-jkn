import re

with open("src/generate_notebook.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Replace df = df_main.copy()
text = text.replace("df = df_main.copy()", "df = df_main")

# 2. Add garbage collection after df_model is created to free RAM
memory_cleanup = """df_model = df[['id'] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()

import gc
try:
    del df, df_diag, df_proc, diag_agg, proc_agg
except:
    pass
gc.collect()
print("\\n[INFO] Memori telah dibersihkan setelah pembuatan df_model.")
"""
text = text.replace("df_model = df[['id'] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()", memory_cleanup)

with open("src/generate_notebook.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Memory optimizations applied!")
