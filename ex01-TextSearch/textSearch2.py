from pyspark.sql import SparkSession
import argparse

def count_errors_in_log(input_path, output_uri, error_keyword="ERROR"):
    spark = SparkSession.builder \
        .appName("LogErrorCounter") \
        .getOrCreate()

    try:
        file_rdd = spark.sparkContext.textFile(input_path)
        errors_rdd = file_rdd.filter(lambda line: error_keyword in line)

        # Contar y transformar los resultados en un DataFrame
        error_count = errors_rdd.count()
        result_df = spark.createDataFrame([(error_count,)], ["error_count"])

        # Escribir el DataFrame en formato Parquet
        result_df.write.mode("overwrite").parquet(output_uri)

    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count ERROR lines in log file')
    parser.add_argument('--data_source', required=True, help='Input file path (e.g., s3://bucket/path/to/file.log)')
    parser.add_argument('--output_uri', required=True, help='Output URI (e.g., s3://bucket/path/to/output)')
    args = parser.parse_args()

    count_errors_in_log(args.data_source, args.output_uri)

