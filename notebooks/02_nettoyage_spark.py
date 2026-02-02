#====================================================================================================
#                              Etape 2 - Nettoyage Spark et agrégation
#                   Objectif: Transformer les données brutes en données exploitables
#====================================================================================================
import os
import sys
from pathlib import Path
import platform
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType

#-------------------------------------------------------------------------------
# Config. des variables d'environement (Python, Java, Hadoop & Pyspark)
#-------------------------------------------------------------------------------
# ───── Python ───────────────
PYTHON_PATH = r"C:\Users\Administrateur\Desktop\spark-pandas-viz-lab\.venv\Scripts\python.exe"
#PYTHON_PATH = r"C:\Users\joel\Desktop\spark-pandas-viz-lab\.venv\Scripts\python.exe"
# ───── Java ─────────────────
os.environ["JAVA_HOME"] = r"C:\Users\Administrateur\data-info-m2i\tools-installation\languages\java\java11"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
# ───── Hadoop ───────────────
os.environ["HADOOP_HOME"] = r"C:\Users\Administrateur\data-info-m2i\tools-installation\app-big-data\hadoop"
os.environ["PATH"] += os.pathsep + r"C:\Users\Administrateur\data-info-m2i\tools-installation\app-big-data\hadoop\bin"
#os.environ["HADOOP_HOME"] = r"C:\Users\joel\data-info\dev-tools\data-tools\hadoop"
#os.environ["PATH"] += os.pathsep + r"C:\Users\joel\data-info\dev-tools\data-tools\hadoop\bin"
# ───── Pyspark ──────────────
os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH

#-------------------------------------------------------------------------------
# Config. des Chemins des données
#-------------------------------------------------------------------------------
DATA_DIR = (Path.cwd() / "." / "data").resolve()
OUTPUT_DIR = (Path(__file__).resolve().parent / ".." / "data" / "output" / "air_quality_clean").resolve()
AIR_QUALITY_PATH = os.path.join(DATA_DIR, "air_quality_raw.csv")
STATIONS_PATH = os.path.join(DATA_DIR, "stations.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "weather_raw.csv")

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def show_startup_message():
    print("=" * 100)
    print(" " * 30 + "Etape 2 - Nettoyage Spark et agrégation")
    print(" " * 20 + "Objectif: Transformer les données brutes en données exploitables")
    print("=" * 100)

def create_spark_session() -> SparkSession | None:
    """Cree et configure la session Spark."""
    spark = SparkSession.builder \
        .appName("TP Qualite Air - Nettoyage") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    ## Reduire les logs
    spark.sparkContext.setLogLevel("ERROR")

    ## Affichage de l'actuelle version de Spark & de l'url de Spark UI
    print()
    print(f"- Spark version: {spark.version}")
    print(f"- Spark UI: {spark.sparkContext.uiWebUrl}")

    ## Infos sur hadoop
    print()
    print("- Hadoop is loaded :", spark.sparkContext._jvm.org.apache.hadoop.io.nativeio.NativeIO.isAvailable())
    print("- Version de Hadoop utilisee par spark :", spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
    print("- HADOOP_HOME :", spark.sparkContext._jvm.java.lang.System.getenv("HADOOP_HOME"))

    ## Info sur python
    print()
    print("- Python version :", platform.python_version())
    print("- Python path :", sys.executable)

    ## Info sur java
    print()
    print("- Java version :", spark._jvm.java.lang.System.getProperty("java.version"))
    print("- JAVA_HOME :", spark._jvm.java.lang.System.getProperty("java.home"))
    print()

    ## Affichage du message du succès
    print("[ok]: creation de la session Spark avec succes.")

    return spark

def stop_spark_session(spark: SparkSession):
    """Stop Spark session"""
    spark.stop()
    # Force kill the Py4J gateway (Windows-safe)
    spark.sparkContext._gateway.shutdown()
    print("─" * 100)
    print("[info]: entire spark session is stopped successfully.")
    print("─" * 100)
    os._exit(0)

def parse_multi_format_timestamp(timestamp_str) -> datetime | None:
    """
    UDF pour parser les timestamps multi-formats.
    Formats supportes:
    - %Y-%m-%d %H:%M:%S (ISO)
    - %d/%m/%Y %H:%M (FR)
    - %m/%d/%Y %H:%M:%S (US)
    - %Y-%m-%dT%H:%M:%S (ISO avec T)
    """
    if timestamp_str is None:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    return None

def clean_value(value_str):
    """
    UDF pour nettoyer les valeurs numeriques.
    - Remplace la virgule par un point
    - Retourne None pour les valeurs non numeriques
    """
    if value_str is None:
        return None

    try:
        # Remplacer virgule par point
        clean_str = value_str.replace(",", ".")
        return float(clean_str)
    except (ValueError, AttributeError):
        return None

def close_spark_session(spark:SparkSession):
    ## Release resources 
    spark.stop()
    # Force kill the Py4J gateway (Windows-safe)
    spark.sparkContext._gateway.shutdown()
    print("─" * 100)
    print("[info]: la session spark a ete arretee avec succes.")
    print("─" * 100)
    os._exit(0)

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def main():
    try:
        ## Clear console
        clear_console()

        ## Show startup message
        show_startup_message()

        ## Creation d'une session Spark 
        print("\nCreation d'une session Spark ...\n")
        spark = create_spark_session() 

        ## Enregistrer les UDFs
        parse_timestamp_udf = F.udf(parse_multi_format_timestamp, TimestampType())
        clean_value_udf = F.udf(clean_value, DoubleType())

        ## Charger les donnees brutes 
        print("─" * 100)
        print("[1/6] Chargement des donnees brutes ...\n")

        df_air_raw = spark.read \
            .option("header", "true") \
            .csv(AIR_QUALITY_PATH)
        
        initial_count = df_air_raw.count()
        print(f"    - Lignes en entree : {initial_count:,}")
        print(f"    - Nombre de colonnes : {len(df_air_raw.columns)}")
        print("    - Schema :")
        df_air_raw.printSchema()
        print("    - Appercu des données :")
        df_air_raw.show(10, truncate=False)
        
        ## Charger les stations pour avoir la capacite
        df_stations = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(STATIONS_PATH)
        
        # ----------------------------------------------------------------
        #               Nettoyage: Traitement des timestamps
        # ----------------------------------------------------------------
        # - Formats de dates multiples => parsing + suppression des timestamps nulles

        print("─" * 100)
        print("[2/6] Traitement des timestamps multi-formats (a partir de la colonne 'timestamp') ...\n")

        df_with_timestamp = df_air_raw.withColumn(
            "timestamp_parsed",
            parse_timestamp_udf(F.col("timestamp"))
        )

        ## Filtrer les timestamps nulles (apres le parsing)
        invalid_timestamps = df_with_timestamp.filter(F.col("timestamp_parsed").isNull()).count()
        df_with_timestamp = df_with_timestamp.filter(F.col("timestamp_parsed").isNotNull())
        print(f"- Timestamps invalides supprimes: {invalid_timestamps:,}")
        
        ## Affichage d'infos après parsing des timestamp
        print("- Nouveau Schema:")
        df_with_timestamp.printSchema()
        print("- Apercu des données :")
        df_with_timestamp.show()

        # ----------------------------------------------------------------
        #                Nettoyage: Traitement des valeurs
        # ----------------------------------------------------------------
        # - Valeurs non numeriques & nulles => Supression 
        # - Valeurs avec virgule            => Remplace la virgule decimal par un point (udf: clean_value_udf)
        # - Valeurs negatives               => Supression
        # - Outliers (>1000 ug/m3)          => Supression

        print("─" * 100)
        print("[3/6] Traitement des valeurs (a partir de la colonne 'value') ...\n")

        ## Remplace la virgule decimal par un point
        df_with_values = df_with_timestamp.withColumn(
            "value_clean",
            clean_value_udf(F.col("value"))
        )
        # df_with_values.show()
        
        ## Filtrer les valeurs non numeriques => transformées en Null grace à l'udf 'clean_value_udf'
        
        ## Filtrer les valeurs nulles
        non_numeric_values = df_with_values.filter(
            F.col("value_clean").isNull()
        ).count()

        df_with_values = df_with_values.filter(
            F.col("value_clean").isNotNull()
        )

        print(f"- Valeurs non numeriques supprimees : {non_numeric_values:,}")

        ## Supprimer les valeurs négatives et les outliers (>1000 ug/m3)
        print("─" * 100)
        print("[4/6] Suppression des valeurs negatives et les outliers (aberrantes > 1000 µ/m3) ...\n")

        negative_count = df_with_values.filter(F.col("value_clean") < 0).count()
        outlier_count = df_with_values.filter(F.col("value_clean") > 1000).count()

        df_clean = df_with_values.filter(
            (F.col("value_clean") >= 0) & (F.col("value_clean") <= 1000)
        )
        print(f"- Nombre des valeurs negatives supprimees : {negative_count:,}")
        print(f"- Nombre des outliers (>1000) supprimes : {outlier_count:,}")

        # df_clean.show()

        ## Dédupliquer sur `(station_id, timestamp, pollutant)`
        print("─" * 100)
        print("[5/6] Deduplication ...\n")
        before_dedup = df_clean.count()
        df_dedup = df_clean.dropDuplicates(["station_id", "timestamp_parsed", "pollutant"])
        after_dedup = df_dedup.count()
        duplicates_removed = before_dedup - after_dedup
        print(f"- Doublons supprimes : {duplicates_removed:,}")

        ## Calculer les moyennes horaires par station et polluant
        print("─" * 100)
        print("[6/6] Agregation horaire et sauvegarde ...\n")
        ## Ajouter les colonnes de temps
        df_with_time = df_dedup.withColumn(
            "date", F.to_date(F.col("timestamp_parsed"))
        ).withColumn(
            "hour", F.hour(F.col("timestamp_parsed"))
        ).withColumn(
            "year", F.year(F.col("timestamp_parsed"))
        ).withColumn(
            "month", F.month(F.col("timestamp_parsed"))
        )
        # print("df_with_time : ")
        # df_with_time.show()

        ## Agreger par heure
        df_hourly = df_with_time.groupBy(
            "station_id", "pollutant", "unit", "date", "hour", "year", "month"
        ).agg(
            F.round(F.mean("value_clean"), 2).alias("value_mean"),
            F.round(F.min("value_clean"), 2).alias("value_min"),
            F.round(F.max("value_clean"), 2).alias("value_max"),
            F.count("*").alias("measurement_count")
        )
        # print("df_hourly : ")
        # df_hourly.show()
        
        ## Joindre avec les informations des stations
        df_final = df_hourly.join(
            df_stations.select("station_id", "station_name", "city", "station_type"),
            on="station_id",
            how="left"
        )
        # print("df_final : ")
        # df_final.show()

        ## Sauvegarder en Parquet partitionné par `date`
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        df_final.write \
                .mode("overwrite") \
                .partitionBy("date") \
                .parquet(str(OUTPUT_DIR))

        ## Rapport : lignes en entrée, lignes supprimées, lignes en sortie
        final_count = df_final.count()

        ## Rapport final
        print("RAPPORT DE NETTOYAGE")
        print(f"Lignes en entree:              {initial_count:>12,}")
        print(f"Timestamps invalides:          {invalid_timestamps:>12,}")
        print(f"Valeurs non numeriques:        {non_numeric_values:>12,}")
        print(f"Valeurs negatives:             {negative_count:>12,}")
        print(f"Outliers (>1000):              {outlier_count:>12,}")
        print(f"Doublons:                      {duplicates_removed:>12,}")
        total_removed = invalid_timestamps + non_numeric_values + negative_count + outlier_count + duplicates_removed
        print(f"Total lignes supprimees:       {total_removed:>12,}")
        print(f"Lignes apres agregation:       {final_count:>12,}")
        print(f"\nFichiers Parquet sauvegardes dans: {OUTPUT_DIR}")

        ## Afficher un apercu
        print("\nApercu des donnees nettoyees:")
        df_final.show(10)

        ## Statistiques par polluant
        print("\nStatistiques par polluant:")
        df_final.groupBy("pollutant") \
            .agg(
                F.count("*").alias("records"),
                F.round(F.mean("value_mean"), 2).alias("avg_value"),
                F.round(F.min("value_min"), 2).alias("min_value"),
                F.round(F.max("value_max"), 2).alias("max_value")
            ) \
            .orderBy("pollutant") \
            .show()

        print("─" * 100)
        print("[ok]: le script a été exécuté avec succès.")
        print("─" * 100)

    except Exception as e:
        print("─" * 100)
        print(f"[ko]: {e}")
        print("─" * 100)
        
    finally:
        ## Fermer Spark
        if spark:
            close_spark_session(spark)



if __name__ == "__main__":
    main()
