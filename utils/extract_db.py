import sys
import os
import s3fs

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pendulum


def get_filesystem():
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    return fs


def sample_data(df_path: str, n_lines: str, time_window_month: str):
    fs = get_filesystem()
    # Charger le DataFrame depuis le fichier Parquet
    df = (
        ds.dataset(
            f"{df_path}",
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .to_pandas()
    )
    # Convertir la colonne de dates en format datetime si ce n'est pas déjà fait
    df["date_modification_dt"] = pd.to_datetime(df["date_modification"])
    # Filtrer les lignes pour avoir 1 mois glissant
    # Get today's date (better to instantiate)
    # Set the local timezone explicitly
    local_tz = pendulum.timezone('Europe/Paris')
    # Get today's date in the local timezone
    today = pendulum.now(local_tz)
    # Calculate the last month's date with the same day
    time_window_month = int(time_window_month)
    last_month_date = today.subtract(months=time_window_month)
    df = df[df["date_modification_dt"] >= last_month_date]
    # Extraire n lignes au hasard uniformément
    n = int(n_lines)  # Remplacez 10 par le nombre de lignes que vous souhaitez extraire
    random_rows = df.sample(n)
    # Récupérer la dernière date disponible dans la table
    last_date = df["date_modification_dt"].max().strftime("%Y%m%d")
    # Supprimer la colonne datetime si elle existe, après traitement (pour traitement ultérieur JSON)
    if "date_modification_dt" in random_rows.columns:
        random_rows = random_rows.drop("date_modification_dt", axis=1)
    # Sauvegarder le résultat dans un nouveau fichier Parquet
    output_file = f"extrait_random_sirene_last_date_{last_date}.parquet"
    pq.write_table(pa.Table.from_pandas(random_rows), output_file)
 
    print(output_file)


def main(df_path: str, number_of_lines: str, time_window_month: str):  # , date_to_log: str):
    # Define file system
    fs = get_filesystem()
    # List all the files in the prefix folder
    files = fs.ls(df_path)
    # Sort the files based on their modification time (last modified first)
    files_sorted = sorted(files, key=lambda x: fs.info(x)['LastModified'], reverse=True)
    # Get the last file in the sorted list
    last_file = files_sorted[0]
    # Sample data to annotate
    sample_data(last_file, number_of_lines, time_window_month)


if __name__ == "__main__":
    df_path = str(sys.argv[1])
    number_of_lines = str(sys.argv[2])
    time_window_month = str(sys.argv[3])
    main(df_path, number_of_lines, time_window_month)
