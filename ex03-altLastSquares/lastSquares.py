from pyspark.sql import SparkSession
import numpy as np
import argparse
import math

def update_user(user_id, Rb, M):
    """Update the user matrix U given the user ID and ratings R."""
    R = Rb.value
    # Extract ratings for the user_id
    user_ratings = R[user_id]
    # Optimize U for this user
    # Implement your logic for updating U (this is a simple example)
    return np.random.rand(len(user_ratings))  # Dummy implementation

def update_movie(movie_id, Rb, U):
    """Update the movie matrix M given the movie ID and ratings R."""
    R = Rb.value
    # Extract ratings for the movie_id
    movie_ratings = R[:, movie_id]
    # Optimize M for this movie
    # Implement your logic for updating M (this is a simple example)
    return np.random.rand(len(movie_ratings))  # Dummy implementation

def als(input_path, output_uri, iterations, k):
    spark = SparkSession.builder.appName("AlternatingLeastSquares").getOrCreate()
    
    # Cargar las calificaciones (R) desde un archivo
    R = np.loadtxt(input_path, delimiter=',')  # Asumimos que el archivo es una matriz
    Rb = spark.sparkContext.broadcast(R)  # Hacemos broadcast de R

    m, u = R.shape  # Número de películas y usuarios
    M = np.random.rand(m, k)  # Inicializar M
    U = np.random.rand(u, k)  # Inicializar U

    for _ in range(iterations):
        # Actualizar U
        U = spark.sparkContext.parallelize(range(u)).map(lambda j: update_user(j, Rb, M)).collect()
        
        # Actualizar M
        M = spark.sparkContext.parallelize(range(m)).map(lambda j: update_movie(j, Rb, U)).collect()

    # Crear un DataFrame para guardar el resultado
    transformed_df = spark.createDataFrame(U, schema=["user_vector_{}".format(i) for i in range(k)])
    
    # Guardar el DataFrame en formato Parquet
    transformed_df.write.mode("overwrite").parquet(output_uri)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alternating Least Squares (ALS) using Spark')
    parser.add_argument('--data_source', required=True, help='Input file path (e.g., s3://bucket/path/to/ratings.csv)')
    parser.add_argument('--output_uri', required=True, help='Output URI (e.g., s3://bucket/path/to/output)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for ALS')
    parser.add_argument('--k', type=int, default=5, help='Number of latent factors')

    args = parser.parse_args()

    als(args.data_source, args.output_uri, args.iterations, args.k)

