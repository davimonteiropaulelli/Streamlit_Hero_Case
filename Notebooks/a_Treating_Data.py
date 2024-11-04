import pandas as pd

# Carregando os datasets
heroes_info = pd.read_csv(r"C:\Users\dpaul\OneDrive\Documentos\Heroes_Case_Alelo\Dataset\Not_Treated\heroes_information.csv")
hero_powers = pd.read_csv(r"C:\Users\dpaul\OneDrive\Documentos\Heroes_Case_Alelo\Dataset\Not_Treated\super_hero_powers.csv")

# Renomeando a coluna de nome do herói para cruzamento
heroes_info.rename(columns={"name":"hero_names"}, inplace=True)

# Join nas tabelas pela coluna de nome do heroi
df = heroes_info.merge(hero_powers, on = ["hero_names"])
df = df.drop(df.columns[0], axis=1)

# Removendo a coluna skin color que possui boa parte dos dados nulos
print(df["Skin color"].value_counts())
df.drop(columns=["Skin color"], inplace=True)

# Contando os nulos antes de substituir por valores
print("Before null replacement:\n")
print(pd.isnull(df).sum().sort_values(ascending=False))

df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])
df['Weight'] = df['Weight'].fillna(df['Weight'].mean())

# Contando os nulos após substituição por valores
print("After null replacement:\n")
print(pd.isnull(df).sum().sort_values(ascending=False))

# Removendo registros que contenham a string '-'
df = df[~df.apply(lambda x: '-'.strip() in str(x.values), axis=1)]

# Removendo herois neutros
df = df[df["Alignment"] != "neutral"]

df.to_csv(r"C:\Users\dpaul\OneDrive\Documentos\Heroes_Case_Alelo\Dataset\Treated\treated_data.csv")