package com.mt

import ml.dmlc.xgboost4j.scala.XGBoost
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType


/**
  * Created by yihaibo on 2019-08-12.
  */
object TestXgb {
  def main(args: Array[String]): Unit = {
    implicit val sparkSession = SparkSession
      .builder()
      .config(
        new SparkConf()
          .setAppName("xgboost")
          .setMaster("local")
      ).getOrCreate()
    val src = sparkSession.read
      .format("csv")
      .option("header","true")
      .option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ")
      .load("file:///Users/didi/gitlab/xgboost4j-spark/src/main/resources/sample_xw.csv")


    //转为double
    val df = src.select(src.columns.map {
      name => col(name).cast(DoubleType)
    }: _*)
    df.printSchema()
    df.show(false)

    val model = XGBoost.loadModel("file:///Users/didi/gitlab/xgboost4j-spark/src/main/resources/model_v3.model")
    val classification_model = new XGBoostClassificationModel(model)

    val vectorAssembler = new VectorAssembler().
      setInputCols(df.columns).
      setOutputCol("features")
    val xgbInput = vectorAssembler.transform(df).select("features")

    xgbInput.show(false)

    val df2 = classification_model.transform(xgbInput)

    df2.show(false)

    df2.groupBy("prediction").count().show(false)
  }
}
