from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("WineQualityTraining") \
        .getOrCreate()

    # Path to your dataset (update this path)
    file_path = "TrainingDataset.csv"

    # Read the CSV file with incorrect headers
    raw_data = spark.read \
        .option("header", "true") \
        .option("delimiter", ";") \
        .csv(file_path)

    # Get the existing column names
    old_column_names = raw_data.columns

    # Define cleaned column names
    cleaned_column_names = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality"
    ]

    # Rename columns
    for old_name, new_name in zip(old_column_names, cleaned_column_names):
        raw_data = raw_data.withColumnRenamed(old_name, new_name)

    # Convert columns to numeric types
    numeric_columns = cleaned_column_names[:-1]  # All columns except 'quality'

    for column in numeric_columns:
        raw_data = raw_data.withColumn(column, col(column).cast(DoubleType()))

    # Ensure 'quality' is treated as a numeric type (e.g., integer)
    raw_data = raw_data.withColumn("quality", col("quality").cast("integer"))

    # Show cleaned schema and data (optional)
    raw_data.printSchema()
    raw_data.show()

    # Prepare data for training
    feature_columns = numeric_columns

    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(raw_data)

    # Define the model
    lr = LogisticRegression(featuresCol="features", labelCol="quality", maxIter=10)

    # Create a pipeline
    pipeline = Pipeline(stages=[lr])

    # Train the model
    model = pipeline.fit(assembled_data)

    # Save the trained model
    model.write().overwrite().save("WineQualityModel")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()