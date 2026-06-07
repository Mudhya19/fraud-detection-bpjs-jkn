import re
with open('src/generate_notebook.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's completely replace the entire `print(f"... df_main (klaim utama)...` block
# to be safely formatted.
import ast

def fix_print_block(content):
    # Fix the df_main, df_diag, df_proc print block
    content = re.sub(
        r'print\(f"\n?✅ df_main \(klaim utama\).*?\nprint\(f"✅ df_proc \(prosedur\)\s*:\s*\{df_proc\.shape\}"\)\n',
        'print(f"\\\\n✅ df_main (klaim utama)  : {df_main.shape}")\nprint(f"✅ df_diag (diagnosis)    : {df_diag.shape}")\nprint(f"✅ df_proc (prosedur)     : {df_proc.shape}")\n',
        content,
        flags=re.DOTALL
    )
    return content

new_text = fix_print_block(text)

with open('src/generate_notebook.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Replaced!")
