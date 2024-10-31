from pyspark.sql import SparkSession
import numpy as np
import argparse

def update_user(user_id, Rb, M, k):
    """Actualiza el vector de características del usuario dado el ID del usuario y las calificaciones R."""
    R = Rb.value
    user_ratings = R[user_id]
    
    # Calcular el nuevo vector del usuario
    updated_U = np.zeros(k)
    total_weight = 0
    
    for movie_id, rating in enumerate(user_ratings):
        if rating > 0:  # Solo considerar calificaciones positivas
            updated_U += rating * M[movie_id]
            total_weight += rating
            
    if total_weight > 0:
        updated_U /= total_weight  # Promediar según el peso
    
    return updated_U

def update_movie(movie_id, Rb, U, k):
    """Actualiza el vector de características de la película dado el ID de la película y las calificaciones R."""
    R = Rb.value
    movie_ratings = R[:, movie_id]
    
    # Calcular el nuevo vector de la película
    updated_M = np.zeros(k)
    total_weight = 0
    
    for user_id, rating in enumerate(movie_ratings):
        if rating > 0:  # Solo considerar calificaciones positivas
            updated_M += rating * U[user_id]
            total_weight += rating
            
    if total_weight > 0:
        updated_M /= total_weight  # Promediar según el peso
    
    return updated_M

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
        U = spark.sparkContext.parallelize(range(u)).map(lambda j: update_user(j, Rb, M, k)).collect()
        
        # Actualizar M
        M = spark.sparkContext.parallelize(range(m)).map(lambda j: update_movie(j, Rb, U, k)).collect()

    # Crear un DataFrame para guardar el resultado
    user_vectors_df = spark.createDataFrame(U, schema=["user_vector_{}".format(i) for i in range(k)])
    
    # Guardar el DataFrame en formato Parquet
    user_vectors_df.write.mode("overwrite").parquet(output_uri)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alternating Least Squares (ALS) using Spark')
    parser.add_argument('--data_source', required=True, help='Input file path (e.g., s3://bucket/path/to/ratings.csv)')
    parser.add_argument('--output_uri', required=True, help='Output URI (e.g., s3://bucket/path/to/output)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for ALS')
    parser.add_argument('--k', type=int, default=5, help='Number of latent factors')

    args = parser.parse_args()

    als(args.data_source, args.output_uri, args.iterations, args.k)
