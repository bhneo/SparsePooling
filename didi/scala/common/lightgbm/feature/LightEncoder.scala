package com.cxyx.common.lightgbm.feature

import akka.routing.Broadcast
import com.cxyx.common.lib
import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.didichuxing.dm.common.tools.log.IColorText
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import scopt.OptionParser
import tools.ToolsHdfs



object LightEncoder extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()


  override def run(): Unit = {
    import spark.implicits._

    log.info(s"cols to encode: ${param.encodeCols}")

    val modelPath = s"${param.rootPath}/output/TAG=${param.modelTag}/fmap/${param.modelDate}"
    val model = if(ToolsHdfs.checkHdfsFileExists(log, modelPath)) {
      log.info(s"model exists, load it now!")
      log.info(s"path:${modelPath}")
      spark.read.parquet(modelPath)
    } else {
      log.error(s"model not exists, fit it now!")
      generateModel(modelPath)
    }

    log.info(s"model table:")
    model.show(10)

    val featurePath = s"${param.rootPath}/output/TAG=${param.featureTag}/feature/${param.featureDate}"

    log.info(s"read feature from ${featurePath}")
    var feature = spark.read.parquet(featurePath)

    log.info(s"transforming...")
    val encodePath = s"${param.rootPath}/output/TAG=${param.featureTag}/samples/${param.featureDate}"
    param.encodeCols.split(",").foreach(c => {
      log.info(s"processing ${c} ...")
      val kv = model.select(c, s"${c}_idx").dropDuplicates(Seq(c)).cache()
      feature = feature.
        join(kv, Seq(c), "left").
        drop(c).
        withColumnRenamed(s"${c}_idx", c).
        na.
        fill(-1,Seq(c))
      kv.unpersist()
    })

    log.info(s"saving encoded ckp...")
    feature.saveCompressedParquet(s"${encodePath}.ckp")
    val featureEncode = spark.read.parquet(s"${encodePath}.ckp").cache()
    featureEncode.show(10, false)

    log.info(s"merging encoded ckp...")
    var mergeCols = List[String]()
    val dropCols = param.dropCols.split(",")
    feature.columns.foreach(c => {
      if (!dropCols.contains(c)) {
        mergeCols = mergeCols :+ c
      }
    })

    log.info("merge columns into features...")
    val result = featureEncode.withColumn("features",
      concat_ws(",", mergeCols.map(c => col(c).cast(LongType)):_*)).cache()

    result.show(10, false)
    log.info("saving...")
    result.saveCompressedParquet(encodePath)
    result.unpersist()
    ToolsHdfs.delHdfsFile(log, s"${encodePath}.ckp")
    log.info(s"data saved to ${encodePath}")

  }

  def generateModel(modelPath: String): DataFrame = {
    val featureFitPath = s"${param.rootPath}/output/TAG=${param.modelTag}/feature/${param.modelDate}"
    log.info(s"reading fit data from ${featureFitPath} ...")
    var featureFit = spark.read.parquet(featureFitPath).cache()
    param.encodeCols.split(",").foreach(c => {
      log.info(s"processing ${c} ...")
      val indexByCount = featureFit.select(c)
        .na.drop().
        filter(s"${c} <> '' ").
        groupBy(c).count().
        orderBy(desc("count")).
        rdd.map{r =>
          r.getAs[String](c).trim
        }.zipWithIndex()
      log.info(indexByCount.collect().toList.take(10))
      val rowRdd = indexByCount.map(a => Row(a._1, a._2.toLong))
      val schema = StructType(
        Array(
          StructField(c, StringType, true),
          StructField(s"${c}_idx", LongType, true)
        )
      )
      val indexed = spark.createDataFrame(rowRdd, schema)
      log.info(s"join to fit data...")
      featureFit = featureFit.join(indexed, Seq(c), "left")
    })
    featureFit.saveCompressedParquet(modelPath)
    featureFit.unpersist()
    log.info(s"model saved to ${modelPath}")
    spark.read.parquet(modelPath)
  }


  override def getOptParser: OptionParser[Config] = new OptionParser[Config](" ") {
    head("feature")
    opt[String]("featureTag").required().text("featureTag").
      action((x, c) => c.copy(featureTag = x))
    opt[String]("featureDate").required().text("featureDate").
      action((x, c) => c.copy(featureDate = x))
    opt[String]("modelTag").required().text("modelTag").
      action((x, c) => c.copy(modelTag = x))
    opt[String]("modelDate").required().text("modelDate").
      action((x, c) => c.copy(modelDate = x))
    opt[String]("rootPath").required().text("rootPath").
      action((x, c) => c.copy(rootPath = x))
    opt[String]("encodeCols").required().text("encodeCols").
      action((x, c) => c.copy(encodeCols = x))
    opt[String]("dropCols").required().text("dropCols").
      action((x, c) => c.copy(dropCols = x))
    opt[String]("key").required().text("key").
      action((x, c) => c.copy(key = x))

  }

  case class Config(
                     featureTag:String = null,
                     featureDate:String = null,
                     modelTag:String = null,
                     modelDate:String = null,
                     rootPath:String = null,
                     encodeCols:String = null,
                     dropCols:String = null,
                     key:String = null
                   )

}
