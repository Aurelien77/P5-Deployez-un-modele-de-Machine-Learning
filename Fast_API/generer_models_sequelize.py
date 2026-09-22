import pandas as pd

# Charger les en-têtes et types de X et y
df_x = pd.read_csv("X.csv", nrows=100)
df_y = pd.read_csv("y.csv", nrows=100)


def infer_sequelize_type(series):
  if pd.api.types.is_integer_dtype(series):
    return "DataTypes.INTEGER"
  elif pd.api.types.is_float_dtype(series):
    return "DataTypes.FLOAT"
  else:
    return "DataTypes.STRING"


print("// --- MODELE X (Caractéristiques des employés) ---")
print("const X = sequelize.define('X', {")
print("  id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },")
for col in df_x.columns:
  stype = infer_sequelize_type(df_x[col])
  print(f"  {col}: {{ type: {stype} }},")
print("});\n")

print("// --- MODELE Y (Cible / Départ) ---")
print("const Y = sequelize.define('Y', {")
print("  id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },")
for col in df_y.columns:
  stype = infer_sequelize_type(df_y[col])
  print(f"  {col}: {{ type: {stype} }},")
print("});\n")

print("// --- MODELE RESULTATS (Historique des prédictions) ---")
print(
    "const Resultat = sequelize.define('Resultats', {"
    "  id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },"
    "  modeleUtilise: { type: DataTypes.STRING },"
    "  probabilite: { type: DataTypes.FLOAT },"
    "  prediction: { type: DataTypes.INTEGER },"
    "  seuilApplique: { type: DataTypes.FLOAT }"
    "});"
)
print("Resultats.belongsTo(X);")
print("Resultats.belongsTo(Y);")