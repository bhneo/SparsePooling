package com.cxyx.common.lightgbm

import com.didichuxing.dm.common.tools.log.SimpleLogger
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.{Vector => LibVector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


/**
  * Created by didi on 2019/5/15.
  */
object Utils {

  /**
    * xgboost 的格式不太一样，需要转一下
    * 1. rawPrediction 存的是 margins，而不是 probability，和 mllib 的概念不一致
    * 2. rawPrediction <-- probability
    * 3. probability <-- probability[1]，取预测为正的概率
    *
    * @param df
    * @return
    */
  def customPrediction(df: DataFrame): DataFrame = {
    // 取正例的prob
    val posProb = (vector: org.apache.spark.ml.linalg.DenseVector) => vector.toArray(1)
    val castCol = udf(posProb)

    df.withColumn("posProb", castCol(df("probability")))
      .drop("rawPrediction")
      .withColumnRenamed("probability", "rawPrediction")
      .withColumnRenamed("posProb", "probability")
  }

  def customPrediction2(df: DataFrame): DataFrame = {
    // 取正例的prob
    val posProb = (vector: org.apache.spark.ml.linalg.DenseVector) => vector.toArray(1)
    val castCol = udf(posProb)

    df.withColumn("posProb", castCol(df("probability")))
      .withColumn("prediction", castCol(df("rawPrediction")))
      .drop("probability")
      .withColumnRenamed("posProb", "probability")
  }

  /**
    * 输入的feature强制转换为ml的格式
    *
    * @param df
    * @param colName
    * @return
    */
  def caseVector(df: DataFrame, colName: String): DataFrame = {
    require(df.columns.contains(colName), s"cast key not in dataframe's columns, colName=${colName}")

    val cast = (vector: LibVector) => vector.asML

    val castCol = udf(cast)

    // val featureType = df.schema(colName).dataType
    val featureVector = df.first().getAs[Any](colName)
    featureVector match {
      case vec: Vector => df
      case vec: LibVector => {
        val tempColName = s"${colName}_CastFeatureTemp"
        df.withColumn(tempColName, castCol(df(colName)))
          .drop(colName)
          .withColumnRenamed(tempColName, colName)
      }
      case _ => throw new RuntimeException(s"feature type in invalid, colName=${colName} typeName=${featureVector.getClass.getSimpleName}")
    }

  }

  /**
    * 输出预测结果的分位数
    *
    * @param df
    */
  def showProbPercentiles(df: DataFrame): Unit = {
    SimpleLogger.info("begin to check predictions ... ")

    val MAX_STAT_SIZE = 10000000

    val checkDF = df.select("probability").repartition(1000).cache()
    val size = checkDF.count()
    SimpleLogger.info(s"prediction size=${size}")


    val statDF = if (size > MAX_STAT_SIZE) {
      val samplingRate = MAX_STAT_SIZE.toDouble / size.toDouble
      SimpleLogger.info(s"predictions size > MAX_STAT_SIZE=${MAX_STAT_SIZE}, sampling result for stat with rate=${samplingRate}")
      checkDF.sample(withReplacement = false, samplingRate)
    } else {
      checkDF
    }
    statDF.describe("probability")
    val percentiles = List(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9).map(_.formatted("%.1f"))
    val slots = percentiles.mkString(",")
    val percentileDF = statDF.selectExpr(s"percentile_approx(probability, array(${slots}))")
    percentileDF.show(truncate = false)
    percentileDF.first().get(0).asInstanceOf[Seq[Double]]
      .zip(percentiles)
      .foreach { case (prob, step) =>
        println(s"percentile:${step}\t->\tprob:${prob.formatted("%.6f")}")
      }
  }

  /**
    * 输出评估结果，总体的AUC,Top 10,20,40,80的recall,precision,f1,ndcg
    *
    * @param df
    * @param labelCol label列名
    * @param predictionCol prediction列名
    */
  def evaluate(df: DataFrame,
               labelCol: String,
               predictionCol: String,
               topList: List[Integer]): Unit = {
    println(s"----------- sort data by ${predictionCol} -----------")
    val data=df.withColumn(labelCol, col(labelCol).cast(IntegerType))
    val sortedDF = data.orderBy(desc(predictionCol))
    // sortedDF.show()

    /*********************** AUC ***********************/
    val binaryEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelCol)
      .setRawPredictionCol(predictionCol)
      .setMetricName("areaUnderROC")
    val auc = binaryEvaluator.evaluate(sortedDF)
    println(s"AUC: $auc")

    /*********************** Top  ***********************/
    val pos = sortedDF.filter(s"$labelCol == 1").count().toDouble
    // for 循环
    for( top <- topList ){
      val topDF = sortedDF.limit(top)
      val tp = topDF.filter(s"$labelCol == 1").count().toDouble
      val fp = top - tp
      val fn = pos - tp

      val recall = tp/pos
      val precision = tp/top
      val f1 = 2*tp/(2*tp+fp+fn)
//      val ndcg = calNDCGFromDF(data, labelCol, predictionCol, top)

//      println(s"top $top -> recall: $recall, precision: $precision, f1: $f1, NDCG: $ndcg")
      println(s"top $top -> recall: $recall, precision: $precision, f1: $f1")
    }
  }

  /**
    * 计算Top k的NDCG
    *
    * @param df
    * @param labelCol label列名
    * @param predictionCol prediction列名
    * @param k top K
    */
  def calNDCGFromDF(df:DataFrame, labelCol: String, predictionCol: String, k: Int): Double = {
    // 理想 DCG
    var idealDcg: Double= 0
    val sortedByLabel = df.select(labelCol).orderBy(desc(labelCol)).collect()
    val sortedByPrediction = df.select(labelCol, predictionCol).orderBy(desc(predictionCol)).collect()

    for (i <- 0 until k) {
      // 计算累计效益
      val gain= (1 << sortedByLabel(i).getAs[Integer](labelCol)) -1
      // 计算折扣因子
      val discount= 1.0 / (Math.log(i +2) / Math.log(2))
      idealDcg += gain * discount
    }

    if (idealDcg >0) {
      var dcg: Double= 0
      for (i <-0 until k) {
        // 计算累计效益
        val gain= (1 << sortedByPrediction(i).getAs[Integer](labelCol)) -1
        // 计算折扣因子
        val discount= 1.0 / (Math.log(i +2) / Math.log(2))
        dcg += gain * discount
      }
      dcg / idealDcg
    }
    else 0
  }

}
