import re

with open("src/generate_notebook.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the actual newline with \n inside the f-string
new_text = re.sub(r'print\(f"\n✅ df_main', r'print(f"\\n✅ df_main', text)

with open("src/generate_notebook.py", "w", encoding="utf-8") as f:
    f.write(new_text)

print("generate_notebook.py updated!")
