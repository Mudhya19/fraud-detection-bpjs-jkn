with open("src/generate_notebook.py", "r", encoding="utf-8") as f:
    text = f.read()

# Fix model checkpoint path
text = text.replace("'best_bpjs_fraud_model.keras'", "'../models/best_bpjs_fraud_model.keras'")

# Remove redundant flat-directory image saving and ensure clear ones
text = text.replace("plt.savefig('01_class_distribution.png', dpi=150, bbox_inches='tight')", "")
text = text.replace("plt.savefig('02_eda_features.png', dpi=150, bbox_inches='tight')", "")
text = text.replace("plt.savefig('03_learning_curves.png', dpi=150, bbox_inches='tight')", "")
text = text.replace("plt.savefig('04_evaluation_results.png', dpi=150, bbox_inches='tight')", "")
text = text.replace("plt.savefig('05_embedding_importance.png', dpi=150, bbox_inches='tight')", "")
text = text.replace("plt.savefig('06_permutation_importance.png', dpi=150, bbox_inches='tight')", "")

with open("src/generate_notebook.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Output paths fixed!")
