package com.cxyx.common.alchemist

import com.alibaba.fastjson.JSONObject
import com.didichuxing.dm.common.tools.log.IColorText
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}
import tools.ToolsHdfs

import scala.io.Source

/**
  * Created by zhaolei on 2021/4/26.
  */
object DataUtils extends IColorText {

  def initData(spark:SparkSession,
               dataRoot:String,
               trainSuffix:String,
               testSuffix:String,
               fMapHDFS:String,
               pathLocal:String,
               config:JSONObject,
               add_category:Boolean,
               log:Logger): (DataFrame, DataFrame, Seq[String]) = {

    val featuresCol = config.getString("featuresCol")
    val labelCol = config.getString("labelCol")
    val cateCols = if (add_category) config.getString("categoricalSlotNames") else "-"
    val workers = config.getIntValue("workers")
    val featureNames = parseFMap(fMapHDFS, pathLocal, log)

    val trainData:DataFrame = if(trainSuffix.equals("-")) {
      null
    } else {
      var data = readDataset(spark, dataRoot, trainSuffix, log)
      data = formatData(data, featuresCol)
      data = mergeCategoryFeatures(spark, data, cateCols, featuresCol)
      println(s"train count : ${data.count()}")
      data.groupBy(labelCol).count().show()
      data.repartition(workers).cache()
    }

    val testData:DataFrame = if(testSuffix.equals("-")) {
      null
    } else {
      var data = readDataset(spark, dataRoot, testSuffix, log)
      data = formatData(data, featuresCol)
      data = mergeCategoryFeatures(spark, data, cateCols, featuresCol)
      println(s"test count : ${data.count()}")
      data.groupBy(labelCol).count().show()
      data.cache()
    }

    (trainData, testData, featureNames)
  }

  def parseFMap(fMapHDFS:String,
                pathLocal:String,
                log:Logger): Seq[String] ={
    ToolsHdfs.getHdfsFile(log,fMapHDFS, pathLocal + "/fmap")
    val file = Source.fromFile(pathLocal + "/fmap")
    val featureCols = file.getLines().toSeq.map(x=>{
      x.split("\t")(1).trim
    })
    println(s"features from FMap: ${featureCols.mkString(",")}")
    file.close()
    featureCols
  }

  /**
    * 用于兼容其他不使用ID类特征的模型，在训练前根据模型判断是否需要加入ID类特征
    * @param spark
    * @param data
    * @param cateCols
    * @param featureCol
    * @return
    */
  def mergeCategoryFeatures(spark:SparkSession,
                            data:DataFrame,
                            cateCols:String,
                            featureCol:String): DataFrame = {
    import spark.implicits._
    // 离散列
    println(s"categorical columns: ${cateCols}")
    // feature列
    var vecCols: Array[String] = null
    var df = data
    if (cateCols != "-") {
      // 原始列
      val conCols = Array(featureCol)
      vecCols = cateCols.split(",") ++ conCols
      println(s"cast categorical columns to double...")
      cateCols.split(",").foreach(col => {
        df = df.withColumn(col, $"$col".cast(DoubleType))
      })

      val assembler = new VectorAssembler().setInputCols(vecCols).setOutputCol(s"merged_${featureCol}")
      df = assembler.transform(df).drop(featureCol).withColumnRenamed(s"merged_${featureCol}", featureCol)
      println(s"assembled dataframe:")
      df.show(10)
    }
    df
  }

  def formatData(data:DataFrame,
                 featureCol:String="features"
                 ): DataFrame = {

    val toDense = udf((xs: Vector) => new DenseVector(xs.toArray))
    val formattedData = data.withColumn(s"${featureCol}_dense", toDense(col(featureCol))).
      drop(featureCol).
      withColumnRenamed(s"${featureCol}_dense",featureCol)

    formattedData
  }

  def readDataset(spark:SparkSession, dataRoot:String, suffix:String, log:Logger): DataFrame = {
    val path = s"${dataRoot}/${suffix}"
    val trainData = if (ToolsHdfs.checkHdfsFileExists(log, path)) {
      println(s"read data from ${path}")
      spark.read.parquet(path)
    } else {
      throw new RuntimeException(s"${path} not exists!")
    }

    trainData
  }

}
