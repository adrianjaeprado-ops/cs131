"""
CS131 Final Project - Phase 3
Failure rate by drive model, with average power-on-hours (age proxy) and
average capacity, across all 3 quarters (Q3 2025, Q4 2025, Q1 2026).
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, avg, count, sum as spark_sum, rank

BUCKET = "gs://adrianjaeprado-cs131-backblaze"
INPUT_PATH = f"{BUCKET}/data_Q*/*.csv"
OUTPUT_PATH = f"{BUCKET}/output/failure_rate_by_model"

def main():
    spark = SparkSession.builder.appName("HardDriveFailureAnalysis").getOrCreate()

    start = time.time()

    raw = spark.read.csv(INPUT_PATH, header=True, inferSchema=False)

    df = raw.select(
        col("model"),
        col("failure").cast("int").alias("failure"),
        col("smart_9_raw").cast("double").alias("power_on_hours"),
        col("capacity_bytes").cast("long").alias("capacity_bytes"),
    )

    grouped = df.groupBy("model").agg(
        count("*").alias("total_drive_days"),
        spark_sum("failure").alias("total_failures"),
        avg("power_on_hours").alias("avg_power_on_hours"),
        avg("capacity_bytes").alias("avg_capacity_bytes"),
    ).withColumn(
        "failure_rate_pct",
        (col("total_failures") / col("total_drive_days") * 100),
    )

    grouped.cache()

    window_spec = Window.orderBy(col("failure_rate_pct").desc())
    result = grouped.withColumn("reliability_rank", rank().over(window_spec)) \
                     .orderBy("reliability_rank")

    result.write.mode("overwrite").option("header", True).csv(OUTPUT_PATH)

    elapsed = time.time() - start

    print(f"=== JOB COMPLETE in {elapsed:.2f} seconds ===")
    result.show(20, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
