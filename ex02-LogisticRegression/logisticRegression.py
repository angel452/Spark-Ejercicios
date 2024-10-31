from pyspark.sql import SparkSession
from pyspark import AccumulatorParam
import numpy as np
import argparse
import math

class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return np.zeros_like(value)

    def addInPlace(self, v1, v2):
        return v1 + v2

def parse_point(line):
    """Parse a line of input into a point (x, y)."""
    parts = line.split(',')
    x = np.array([float(i) for i in parts[:-1]])  # All but last are features
    y = float(parts[-1])  # Last element is the label
    return (x, y)

def logistic_regression(input_path, iterations, learning_rate):
    spark = SparkSession.builder.appName("LogisticRegression").getOrCreate()
    
    # Cargar y parsear los puntos desde el archivo
    points_rdd = spark.sparkContext.textFile(input_path).map(parse_point).cache()

    # Dimensiones y pesos
    D = len(points_rdd.first()[0])  # Número de características
    w = np.random.rand(D)  # Inicializar w como un vector aleatorio

    # Crear un acumulador para el gradiente
    grad_accumulator = spark.sparkContext.accumulator(np.zeros(D), VectorAccumulatorParam())

    for _ in range(iterations):
        # Resetear el acumulador en cada iteración
        grad_accumulator.value[:] = 0  # Limpiar el acumulador

        # Calcular el gradiente
        points_rdd.foreach(lambda p: grad_accumulator.add((1 / (1 + math.exp(-p[1] * np.dot(w, p[0])))) - 1) * p[1] * p[0])

        # Actualizar los pesos
        w -= learning_rate * grad_accumulator.value

    return w

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic Regression using Spark')
    parser.add_argument('--data_source', required=True, help='Input file path (e.g., s3://bucket/path/to/data.csv)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for gradient descent')

    args = parser.parse_args()

    final_weights = logistic_regression(args.data_source, args.iterations, args.learning_rate)
    print(f"Final weights: {final_weights}")

