import json

with open("src/generate_notebook.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace diagnosis columns
text = text.replace("kd_diagnosis", "diag")
text = text.replace("level_diagnosis", "levelid")

# Replace procedure columns
text = text.replace("kd_procedure", "proc")

with open("src/generate_notebook.py", "w", encoding="utf-8") as f:
    f.write(text)

# Also fix the notebook directly
try:
    with open("notebooks/bpjs_fraud_detection.ipynb", "r", encoding="utf-8") as f:
        nb_str = f.read()
    nb_str = nb_str.replace("kd_diagnosis", "diag")
    nb_str = nb_str.replace("level_diagnosis", "levelid")
    nb_str = nb_str.replace("kd_procedure", "proc")
    
    with open("notebooks/bpjs_fraud_detection.ipynb", "w", encoding="utf-8") as f:
        f.write(nb_str)
except Exception as e:
    pass

print("Columns fixed!")
