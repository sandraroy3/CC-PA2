from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import sys

def main(test_file_path):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("WineQualityPrediction") \
        .getOrCreate()

    # Path to the trained model
    model_path = "WineQualityModel"

    # Load the trained model
    model = PipelineModel.load(model_path)

    # Load the test data
    test_data = spark.read \
        .option("header", "true") \
        .option("delimiter", ";") \
        .csv(test_file_path)

    # Define column names and rename them
    old_column_names = test_data.columns
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
        test_data = test_data.withColumnRenamed(old_name, new_name)

    # Convert columns to numeric types
    numeric_columns = cleaned_column_names[:-1]  # All columns except 'quality'

    for column in numeric_columns:
        test_data = test_data.withColumn(column, col(column).cast(DoubleType()))

    # Ensure 'quality' is treated as a numeric type
    test_data = test_data.withColumn("quality", col("quality").cast("integer"))

    # Prepare the data (apply feature transformation)
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
    assembled_data = assembler.transform(test_data)

    # Make predictions
    predictions = model.transform(assembled_data)

    # Evaluate the model using F1 Score
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality",
        predictionCol="prediction",
        metricName="f1"
    )

    f1_score = evaluator.evaluate(predictions)
    print(f"F1 Score: {f1_score}")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: predict_model.py <test_file_path>")
        sys.exit(-1)

    test_file_path = sys.argv[1]
    main(test_file_path)