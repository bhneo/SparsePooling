package com.cxyx.common.alchemist

import com.alibaba.fastjson.JSONObject
import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.microsoft.ml.spark.lightgbm.LightGBMClassificationModel
import com.microsoft.ml.spark.lightgbm.LightGBMRankerModel
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.ml.classification.{LogisticRegressionModel, LogisticRegression}
import org.apache.log4j.Logger
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.{Vector => LibVector}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StringType}

import tools.ToolsHdfs

import java.io.PrintWriter
import scala.io.Source
import scala.collection.mutable

/**
  * Created by zhaolei on 2021/4/26.
  */
object EvaluateUtils extends Constant {

  def explainModel(model:Object,
                   fMapHDFS:String,
                   pathLocal:String,
                   modelConfig:JSONObject,
                   addCategory:Boolean,
                   log:Logger): Unit = {
    println(s"get feature map from: $fMapHDFS")
    ToolsHdfs.getHdfsFile(log, fMapHDFS, s"${pathLocal}/fmap")
    model match {
      case m: XGBoostClassificationModel =>
        explainXGB(m,
          pathLocal,
          if (addCategory) modelConfig.getString(CATE_NAME) else "-")
      case m: LightGBMClassificationModel =>
        explainLGB(m,
          pathLocal,
          if (addCategory) modelConfig.getString(CATE_NAME) else "-")
      case m: LightGBMRankerModel =>
        explainLGB(m,
          pathLocal,
          if (addCategory) modelConfig.getString(CATE_NAME) else "-")
      case _ =>
        println(s"model not support for feature importance!")
    }
  }

  def explainXGB(model: XGBoostClassificationModel,
                 pathLocal: String,
                 categoricalSlotNames: String): Unit = {

    println(s"model loaded, n(feature)=${model.numFeatures}")
    val pathFeatureScore = s"$pathLocal/fscore"
    val fvs = model.nativeBooster.getFeatureScore(s"$pathLocal/fmap").toMap
      .toSeq.sortBy(_._2).reverse.map(x=>s"${x._1}\t${x._2}").mkString("\n")
    val file = new PrintWriter(pathFeatureScore)
    file.println(fvs)
    file.close()
    println(s"save fscore to : $pathFeatureScore")

    println("--------------- topK feature score")
    val source = Source.fromFile(pathFeatureScore, "utf-8")
    source.getLines()
      .toSeq
      .slice(0, 100)
      .zipWithIndex
      .foreach(x => println(s"${x._2}\t${x._1}"))
    source.close()
  }

  def explainLGB(model: Object,
                 pathLocal: String,
                 categoricalSlotNames: String): Unit = {
    val _model = model match {
      case m: LightGBMClassificationModel =>
        m
      case m: LightGBMRankerModel =>
        m
      case _ =>
        throw new RuntimeException(s"model not support!")
    }
    println(s"model loaded, n(feature)=${_model.numFeatures}")
    for (x <- List("split", "gain")) {
      val source = Source.fromFile(s"$pathLocal/fmap")
      val names = source.getLines().toSeq
        .map(x => {
          x.split("\\t")(1)
        }).toArray
      source.close()

      var cols: Array[String] = null
      if (categoricalSlotNames != "-") {
        cols = Array.concat(categoricalSlotNames.split(","), names)
      } else {
        cols = names
      }
      val fvs = cols.zip(_model.getFeatureImportances(s"${x}")).toMap
        .toSeq.sortBy(_._2).reverse.map(x => s"${x._1}\t${x._2}").mkString("\n")

      val pathFeatureScore = s"$pathLocal/fscore_${x}"
      val file = new PrintWriter(pathFeatureScore)
      file.println(fvs)
      file.close()
      println(s"save split feature score to : $pathFeatureScore")

      println("--------------- topK feature score")
      val scores = Source.fromFile(pathFeatureScore, "utf-8")
      scores.getLines().toSeq
        .slice(0, 100).zipWithIndex
        .foreach(x => println(s"${x._2}\t${x._1}"))
      scores.close()
    }
  }

  def customPrediction(df: DataFrame, predCol:String, probCol:String): DataFrame = {
    val posProb = (vector: DenseVector) => vector.toArray(1)
    val castCol = udf(posProb)

    df.withColumn("posProb", castCol(df("probability")))
      .withColumn(predCol, castCol(df("rawPrediction")))
      .drop("probability")
      .withColumnRenamed("posProb", probCol)
  }

  def evaluate(spark: SparkSession,
               model: Object,
               trainDF: DataFrame,
               testDF: DataFrame,
               dataRoot: String,
               trainSuffix: String,
               testSuffix: String,
               repredict: Boolean,
               log: Logger,
               config: JSONObject,
               predCol: String="predict",
               probCol: String="prob"
              ): Unit = {

    val featuresCol = config.getString("featuresCol")
    val labelCol = config.getString("labelCol")

    val leafCol = if(config.containsKey("leaf") && !config.getString("leaf").equals("-")) {
      config.getString("leaf")
    } else {
      ""
    }

    val _model = model match {
      case m: XGBoostClassificationModel =>
        if(leafCol.nonEmpty) {
          m.setLeafPredictionCol(leafCol)
        } else {
          m
        }
      case m: LightGBMClassificationModel =>
        if(leafCol.nonEmpty) {
          m.setLeafPredictionCol(leafCol)
        } else {
          m
        }
      case m: LightGBMRankerModel =>
        if(leafCol.nonEmpty) {
          m.setLeafPredictionCol(leafCol)
        } else {
          m
        }
      case m: LogisticRegressionModel =>
        m
      case _ =>
        throw new RuntimeException(s"model not support!")
    }

    val groupCols: Seq[String] = if(config.containsKey("eval_groups")) {
      config.getString("eval_groups").split(",")
    } else {
      Seq("user_id")
    }

    val _predCol = if (_model.isInstanceOf[LightGBMRankerModel]) {
      "prediction"
    } else {
      predCol
    }
    if(trainDF != null) {
      if (repredict || !ToolsHdfs.checkHdfsFileExists(log, dataRoot + s"/predict/${trainSuffix}")) {
        val predictTrainDF = if(_model.isInstanceOf[LightGBMRankerModel]){
          _model.transform(trainDF)
        } else {
          customPrediction(_model.transform(trainDF), _predCol, probCol)
        }
        predictTrainDF.drop(featuresCol).saveCompressedParquet(dataRoot + s"/predict/${trainSuffix}")
      }
      val predict = spark.read.parquet(dataRoot + s"/predict/${trainSuffix}")

      println("----------- Evaluation on training data -----------")
      val auc = calAUC(predict, _predCol, labelCol)
      println(s"AUC: $auc")
      groupCols.foreach(g => {
        val gAuc = calGroupAUC(predict, g, _predCol, labelCol)
        println(s"GAUC on ${g}: $gAuc")
      })
    }

    if(testDF != null) {
      if (repredict || !ToolsHdfs.checkHdfsFileExists(log, dataRoot + s"/predict/${testSuffix}")) {
        val predictTestDF = if(_model.isInstanceOf[LightGBMRankerModel]){
          _model.transform(testDF)
        } else {
          customPrediction(_model.transform(testDF), predCol, probCol)
        }
        predictTestDF.drop(featuresCol).saveCompressedParquet(dataRoot + s"/predict/${testSuffix}")
      }
      val predict = spark.read.parquet(dataRoot + s"/predict/${testSuffix}")

      println("----------- Evaluation on test data -----------")
      val auc = calAUC(predict, _predCol, labelCol)
      println(s"AUC: $auc")
      groupCols.foreach(g => {
        val gAuc = calGroupAUC(predict, g, _predCol, labelCol)
        println(s"GAUC on ${g}: $gAuc")
      })
    }
  }

  def calAUC(predict:DataFrame, predCol:String, labelCol:String): Double = {
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelCol)
      .setRawPredictionCol(predCol)
      .setMetricName("areaUnderROC")

    val auc = evaluator.evaluate(predict)
    auc
  }

  def calGroupAUC(predict: DataFrame, groupCol: String, predCol: String, labelCol: String): Double = {
    val data = predict
      .select(
        col(groupCol),
        col(labelCol),
        concat_ws(
          ";",
          col(predCol).cast(StringType), col(labelCol).cast(StringType)
        ).alias("prediction_label")
      )
      .groupBy(groupCol)
      .agg(
        collect_list(col("prediction_label")).alias("prediction_labels"),
        count(col(groupCol)).alias("user_count"),
        countDistinct(labelCol).alias("distinct_label")
      )
      .filter(col("distinct_label") === 2)

    val total_count = data.select(col("user_count")).rdd.map(_(0).asInstanceOf[Long]).reduce(_+_).toDouble

    val group_auc = data
      .withColumn("weight", col("user_count").cast(DoubleType)/total_count)
      .drop("user_count", "distinct_label", labelCol)
      .rdd
      .map {
        row =>
          val weight = row.getAs[Double]("weight")
          val scoreAndLabels = row.getAs[mutable.WrappedArray[String]]("prediction_labels").toList
            .map(_.split(";").map(_.toDouble).toArray).map{ case Array(f1,f2) => (f1,f2) }

          aucScore(scoreAndLabels) * weight
      }.reduce(_+_)

    group_auc
  }

  def aucScore(scoreAndLabels: List[(Double, Double)]): Double ={
    val sort_arr = scoreAndLabels.sortBy(_._1)
    var M = 0
    var N = 0
    var rank = 1
    var sum = 0
    for (elem <- sort_arr) {
      val label = elem._2
      if (label == 1.0) {
        M += 1
        sum += rank
      }else if (label == 0.0){
        N += 1
      }
      rank += 1
    }
    var auc = 0.0
    if (M * N != 0){
      auc = (sum - M*(M+1)/2).toDouble / (M * N).toDouble
    }
    auc
  }

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

}
