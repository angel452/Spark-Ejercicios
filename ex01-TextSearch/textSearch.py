from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerField, StringField
import argparse

"""
Cuenta la cantidad de líneas que contienen una palabra clave de error específica en un archivo de registro.

Parámetros:
input_path (str): Ruta al archivo de entrada (puede ser una ruta local o S3)
output_uri (str): Ruta donde guardar los resultados
error_keyword (str): Palabra clave que se buscará en las líneas (predeterminado: "ERROR")
    
Devuelve:
int: Cantidad de líneas que contienen la palabra clave de error
"""

def count_errors_in_log(input_path, output_uri=None, error_keyword="ERROR"):

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("LogErrorCounter") \
        .getOrCreate()
    
    try:
        # Crear un RDD desde el txt
        file_rdd = spark.sparkContext.textFile(input_path)
        
        # Filter lines containing the error keyword
        errors_rdd = file_rdd.filter(lambda line: error_keyword in line)
        
        # Cache the filtered RDD if we plan to reuse it
        cached_errors = errors_rdd.cache()
        
        # Map each line to 1 and reduce to get total count
        error_count = cached_errors.map(lambda x: 1).reduce(lambda a, b: a + b)
        
        # Crear un DataFrame con el resultado
        schema = StructType([
            StructField("error_count", IntegerField(), False),
            StructField("input_file", StringField(), False),
            StructField("error_keyword", StringField(), False)
        ])
        
        # Crear una lista con los resultados
        data = [(error_count, input_path, error_keyword)]
        
        # Crear el DataFrame
        result_df = spark.createDataFrame(data, schema)
        
        # Si se proporciona output_uri, guardar los resultados
        if output_uri:
            result_df.write.mode("overwrite").parquet(output_uri)
        
        return error_count
    
    finally:
        # Good practice to stop SparkSession when done
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count ERROR lines in log file')
    
    parser.add_argument('--data_source', 
                       required=True,
                       help='Input file path (e.g., s3://bucket/path/to/file.log)')
    
    parser.add_argument('--output_uri',
                       required=False,
                       help='Output URI (optional)')
    
    args = parser.parse_args()
    
    try:
        count = count_errors_in_log(args.data_source, args.output_uri)
        print(f"Total number of lines containing ERROR: {count}")
        
        if args.output_uri:
            print(f"Results have been saved to: {args.output_uri}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
