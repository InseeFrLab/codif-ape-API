import os
import sys

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm


def is_naf_code(text: str):
    return text[:4].isdigit() and text[4].isalpha()


def transform_json_to_dataframe(json_dir: str):
    transformed_data = []
    files_only = [item for item in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, item))]
    for filename in tqdm(files_only):
        with open(os.path.join(json_dir, filename), "r") as file:
            data = json.load(file)

        annotation_date = data["task"]["updated_at"]
        # Get data variables
        libelle = data["task"]["data"]["libelle"]
        liasse_numero = data["task"]["data"]["liasse_numero"]
        date_modification = data["task"]["data"]["date_modification"]
        NAF2008_code = data["task"]["data"]["apet_finale"]
        mode_calcul_ape = data["task"]["data"]["mode_calcul_ape"]
        evenement_type = data["task"]["data"]["evenement_type"]
        liasse_type = data["task"]["data"]["liasse_type"]
        activ_surf_et = str(data["task"]["data"]["activ_surf_et"])
        activ_nat_et = data["task"]["data"]["activ_nat_et"]
        activ_nat_lib_et = data["task"]["data"]["activ_nat_lib_et"]
        activ_sec_agri_et = data["task"]["data"]["activ_sec_agri_et"]
        cj = data["task"]["data"]["cj"]

        # Number of skips
        skips = int(data["was_cancelled"])
        # Get annotated data without skips and adjust from UI's bugs
        if len(data["result"]) > 0:
            apet_manual = ""
            commentary = ""
            rating = 0
            # indicator to check if NAF 2025 is selected as choice
            NAF2025_OK = 0
            # Check first if NAF2025 is ok in whole dict
            for result in data["result"]:
                # Retrieve choice result
                if "choices" in result["value"]:
                    choices = result["value"]["choices"]
                    if "Oui" in choices: # order
                        NAF2025_OK = 1
            # Then map dict data
            for result in data["result"]:
                # Retrieve comment
                if "text" in result["value"]:
                    commentary = result["value"]["text"][0]
                # Retrieve choice result
                if "choices" in result["value"]:
                    choices = result["value"]["choices"]
                    if (NAF2025_OK == 1) and is_naf_code(choices[0]):
                        apet_manual = choices[0]
                # Retrieve taxonomy result
                if "taxonomy" in result["value"] and (NAF2025_OK == 0):
                    taxonomy_values = result["value"]["taxonomy"][0][-1]
                    apet_manual = taxonomy_values.replace(".", "")  # delete . in apet_manual
                    apet_manual = apet_manual[:5]
                # Retrieve rating result
                if "rating" in result["value"]:
                    rating = result["value"]["rating"]
                # Check if apet is in comment and fill empty apet (due to LS bug)
                if apet_manual == "" and is_naf_code(commentary):
                    print("Potential LS bug --> NAF code in comment: " + commentary)
                    apet_manual = commentary

        # Créer un dictionnaire pour les données transformées
        transformed_row = {
            "liasse_numero": liasse_numero,
            "libelle": libelle,
            "evenement_type": evenement_type,
            "liasse_type": liasse_type,
            "activ_surf_et": activ_surf_et,
            "activ_nat_et": activ_nat_et,
            "activ_nat_lib_et": activ_nat_lib_et,
            "activ_sec_agri_et": activ_sec_agri_et,
            "cj": cj,
            "date_modification": date_modification,
            "annotation_date": annotation_date,
            "NAF2008_code": NAF2008_code,
            "mode_calcul_ape": mode_calcul_ape,
            "apet_manual": apet_manual,
            "commentary": commentary,
            "rating": rating,
            "skips": skips,
        }

        # Append in list
        transformed_data.append(transformed_row)

    # Convert to Dataframe
    results = pd.DataFrame(transformed_data)
    print("Number of lines: " + str(len(results)))
    # Filter skipped results
    skipped_results = results[results["skips"] != 0]
    # Filter unclassifiable results
    unclassifiable_results = results[(results["apet_manual"].str.match(r'^(I|X)'))]

    # Count skipped and unclassifiable
    print("Number of skips: " + str(len(skipped_results)))
    print("Rate of skips: " + str(len(skipped_results)/len(results)))
    print("Number of unclassifiable: " + str(len(unclassifiable_results)))
    print("Rate of unclassifiable: " + str(len(unclassifiable_results)/len(results)))

    # Keep only unskipped and classifiable annotations
    results = results[results["skips"] == 0]
    results = results[~(results["apet_manual"].str.match(r'^(I|X)'))]
    results = results[results["apet_manual"] != ""]
    print("Number of lines: " + str(len(results)))

    return results, skipped_results, unclassifiable_results


def save_to_s3(table: list[pa.Table], bucket: str, path: str, category: str):
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    pq.write_table(table[0], f"s3://{bucket}/{path}/training_data_{category}_NAF2025.parquet/", filesystem=fs)
    pq.write_table(table[1], f"s3://{bucket}/{path}/skipped_data_{category}_NAF2025.parquet/", filesystem=fs)
    pq.write_table(table[2], f"s3://{bucket}/{path}/unclassifiable_data_{category}_NAF2025.parquet/", filesystem=fs)


def main(annotation_results_path: str, annotation_preprocessed_path: str, category: str):
    # Upload log file from API pod and filter by date
    data, skipped_data, unclassifiable_data = transform_json_to_dataframe(annotation_results_path)

    # Add date column for partitionning
    data["date"] = pd.to_datetime(data["date_modification"]).dt.strftime("%Y-%m-%d")
    skipped_data["date"] = pd.to_datetime(skipped_data["date_modification"]).dt.strftime("%Y-%m-%d")
    unclassifiable_data["date"] = pd.to_datetime(unclassifiable_data["date_modification"]).dt.strftime("%Y-%m-%d")

    data = (
        data[
            [
                "liasse_numero",
                "libelle",
                "evenement_type",
                "liasse_type",
                "activ_surf_et",
                "activ_nat_et",
                "activ_nat_lib_et",
                "activ_sec_agri_et",
                "cj",
                "date",
                "NAF2008_code",
                "mode_calcul_ape",
                "apet_manual",
                "rating"
            ]
        ]
        .reset_index(drop=True)
        #.rename(
        #    columns={
        #        "libelle": "text_description",
        #        "liasse_type": "type_",
        #        "activ_nat_et": "nature",
        #        "activ_surf_et": "surface",
        #        "evenement_type": "event",
        #    }
        #)
        #.reset_index(drop=True)
    )

    # data['surface'] = data['surface'].astype(bytes)
    # Translate pd.DataFrame into pa.Table
    arrow_table = pa.Table.from_pandas(data)
    arrow_skipped_table = pa.Table.from_pandas(skipped_data)
    arrow_unclassifiable_table = pa.Table.from_pandas(unclassifiable_data)

    # Save logs in a partionned parquet file in s3
    save_to_s3([arrow_table, arrow_skipped_table, arrow_unclassifiable_table], "projet-ape", f"/{annotation_preprocessed_path}/", category)


if __name__ == "__main__":
    annotation_results_path = str(sys.argv[1])
    annotation_preprocessed_path = str(sys.argv[2])
    category = str(sys.argv[3])

    main(annotation_results_path, annotation_preprocessed_path, category)
