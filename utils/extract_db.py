import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def sample_data(df_path: str, n_lines: str):
    # Charger le DataFrame depuis le fichier Parquet
    df = pq.read_table(df_path).to_pandas()

    # Convertir la colonne de dates en format datetime si ce n'est pas déjà fait
    df['date_modification_dt'] = pd.to_datetime(df['date_modification'])

    # Filtrer les lignes à partir du 1er janvier 2023
    df = df[df['date_modification_dt'] >= '2023-01-01']

    # Extraire n lignes au hasard uniformément
    n = int(n_lines)  # Remplacez 10 par le nombre de lignes que vous souhaitez extraire
    random_rows = df.sample(n)

    # Récupérer la dernière date disponible dans la table
    last_date = df['date_modification_dt'].max().strftime("%Y%m%d")

    # Supprimer la colonne datetime si elle existe, après traitement (pour traitement ultérieur JSON)
    if 'date_modification_dt' in random_rows.columns:
        random_rows = random_rows.drop('date_modification_dt', axis=1)

    # Sauvegarder le résultat dans un nouveau fichier Parquet
    output_file = f'extrait_random_sirene_last_date_{last_date}.parquet'
    pq.write_table(pa.Table.from_pandas(random_rows), output_file)
    print(output_file)


if __name__ == "__main__":
    df_path = str(sys.argv[1])
    number_of_lines = str(sys.argv[2])
    sample_data(df_path, number_of_lines)
