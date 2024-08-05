import sys
import os
import s3fs

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.model_selection import train_test_split


def get_filesystem():
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    return fs


def sample_data(df_path: str, n_lines: str):
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
    # Calculer la taille de l'échantillon pour chaque strate
    n = int(n_lines)

    # Sélectionner les lignes où categorie_demande est "CG" pour la stratification
    # Exclure celles dont `apet_finale` commence par `01`, `02`, ou `03`
    df_stratify_cg = df[(df['categorie_demande'] == 'CG') & (~df['apet_finale'].str.match(r'^(01|02|03)'))]
    
    # Récupérer les lignes où categorie_demande n'est pas "AGRI" et apet_finale commence par `01`, `02`, ou `03`
    df_cg_apet_010203 = df[(df['categorie_demande'] != 'AGRI') & (df['apet_finale'].str.match(r'^(01|02|03)'))]

    # Ajouter ces lignes à la partie AGRI
    df.loc[df_cg_apet_010203.index, 'categorie_demande'] = 'AGRI'

    # Calculer la taille de chaque strate pour les lignes CG
    total_size_cg = len(df_stratify_cg)
    stratified_sample_dfs = []

    # Échantillonner stratifié pour les lignes CG
    for name, group in df_stratify_cg.groupby('apet_finale'):
        frac = len(group) / total_size_cg
        sample_size = max(1, int(frac * n))
        stratified_sample_dfs.append(group.sample(sample_size, random_state=42))

    # Combiner les échantillons de chaque strate pour les lignes CG
    stratified_sample_df_cg = pd.concat(stratified_sample_dfs)

    # Ajuster l'échantillon pour qu'il soit exactement de taille n
    if len(stratified_sample_df_cg) > n:
        stratified_sample_df = stratified_sample_df_cg .sample(n, random_state=42)
    elif len(stratified_sample_df_cg) < n:
        additional_sample = df.drop(stratified_sample_df_cg.index).sample(n - len(stratified_sample_df_cg), random_state=42)
        stratified_sample_df_cg  = pd.concat([stratified_sample_df_cg , additional_sample])

    # Filtrer les lignes où categorie_demande n'est pas "CG" pour concaténer
    df_concatenate = df[df['categorie_demande'] != 'CG']

    # Concaténer les lignes avec l'échantillon stratifié pour les lignes CG
    stratified_sample_df = pd.concat([stratified_sample_df_cg, df_concatenate])

    print(len(stratified_sample_df))

    # Récupérer la dernière date disponible dans la table
    last_date = df['date_modification_dt'].max().strftime("%Y%m%d")

    # Supprimer la colonne datetime si elle existe, après traitement (pour traitement ultérieur JSON)
    if 'date_modification_dt' in stratified_sample_df.columns:
        stratified_sample = stratified_sample_df.drop('date_modification_dt', axis=1)

    # Partitionner par 'emetteur' et sauvegarder chaque partition dans un fichier Parquet
    for categorie, partition_df in stratified_sample.groupby('categorie_demande'):
        partition_file = os.path.join(f's3://projet-ape/label-studio/annotation-campaign-2024/rev-NAF2025/{categorie}/data-samples/queue/extrait_{categorie}_sirene_last_date_{last_date}.parquet')
        pq.write_table(pa.Table.from_pandas(partition_df), partition_file, filesystem=fs)
        print(f'Saved {partition_file}')


def main(df_path: str, number_of_lines: str):  # , date_to_log: str):
    # Define file system
    fs = get_filesystem()
    # List all the files in the prefix folder
    files = fs.ls(df_path)
    # Sort the files based on their modification time (last modified first)
    files_sorted = sorted(files, key=lambda x: fs.info(x)['LastModified'], reverse=True)
    # Get the last file in the sorted list
    last_file = files_sorted[0]
    # Sample data to annotate
    sample_data(last_file, number_of_lines)


if __name__ == "__main__":
    df_path = str(sys.argv[1])
    number_of_lines = str(sys.argv[2])
    main(df_path, number_of_lines)
