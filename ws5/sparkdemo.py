import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def main():
    # A1. Create a SparkSession named ws5-regression
    spark = SparkSession.builder \
        .appName("ws5-regression") \
        .getOrCreate()

    # A2. Read the dataset from your bucket with header and schema inference
    # The bucket path is passed in as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: sparkdemo.py <gcs-path-to-tips.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"Reading data from: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df.show(5)


    # A3. Combine total_bill and size into a single vector column called features
    assembler = VectorAssembler(
        inputCols=["total_bill", "size"], 
        outputCol="features"
    )

    # A4. Split data into 80% train / 20% test using a fixed seed
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # A5. Define LinearRegression and fit using a Pipeline
    lr = LinearRegression(featuresCol="features", labelCol="tip")
    pipeline = Pipeline(stages=[assembler, lr])
    pipeline_model = pipeline.fit(train_df)

    # A6. Apply the fitted pipeline to the test set to produce predictions
    predictions = pipeline_model.transform(test_df)

    # A7. Evaluate predictions on RMSE and R2
    evaluator_rmse = RegressionEvaluator(labelCol="tip", predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)

    evaluator_r2 = RegressionEvaluator(labelCol="tip", predictionCol="prediction", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)

    # A8. Pull the LinearRegression model out and print metrics
    lr_model = pipeline_model.stages[-1]
    
    print("\n" + "="*40)
    print("RESULTS:")
    print(f"Coefficients: {lr_model.coefficients}")
    print(f"Intercept: {lr_model.intercept}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print("="*40 + "\n")

    spark.stop()

if __name__ == "__main__":
    main()