package com.cxyx.common.lightgbm.feature

import com.cxyx.common.lib
import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.cxyx.common.alchemist.tools.ToolsHdfs
import com.didichuxing.dm.common.tools.log.IColorText
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import scopt.OptionParser




object LightEncoder2 extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()

  override def run(): Unit = {

    log.info(s"cols to encode: ${param.encodeCols}")

    val featurePath = s"${param.rootPath}/output/TAG=${param.featureTag}/feature/${param.featureDate}"
    log.info(s"read feature from ${featurePath}")
    var feature = spark.read.parquet(featurePath)

    log.info(s"transforming...")
    param.encodeCols.split(",").foreach(c => {
      log.info(s"processing ${c} ...")
      val modelPath = s"${param.rootPath}/output/TAG=${param.modelTag}/fmap/${param.modelDate}/${c}"

      if(ToolsHdfs.checkHdfsFileExists(log, modelPath)) {
        log.info(s"old model exists, load it now!")
        log.info(s"path:${modelPath}")
        val oldKV = spark.read.parquet(modelPath)

        val maxIndex = oldKV.selectExpr(s"max(${c}_idx) max_index").collect()(0).getAs[Long]("max_index")
        log.info(s"join old KV to fit data...")
        feature = feature.withColumn(c, trim(col(c))).join(oldKV, Seq(c), "left")
        val count = feature.select(s"${c}_idx").distinct().count()
        log.info(s"old distinct count: ${count}")

        val encoded = feature.filter(s"${c}_idx is not null")
        val encodedCount = encoded.select(s"${c}").distinct().count()
        log.info(s"encoded distinct count: ${encodedCount}")

        val unseen = feature.filter(s"${c}_idx is null").drop(s"${c}_idx")
        val unseenCount = unseen.select(s"${c}").distinct().count()
        log.info(s"unseen distinct count: ${unseenCount}")
        val newKV = makeIndex(unseen, c, maxIndex+1)
        log.info(s"join new KV to fit data...")

        var newModel = newKV.unionByName(oldKV)
        log.info(s"new model key count: ${newModel.count()}")
        newModel = newModel.dropDuplicates(c)
        log.info(s"new model unique key count: ${newModel.count()}")
        val newModelPath = s"${param.rootPath}/output/TAG=${param.modelTag}/fmap/${param.featureDate}/${c}"
        newModel.saveCompressedParquet(newModelPath)
        log.info(s"new model saved to ${newModelPath}")

        feature = unseen.join(newModel, Seq(c), "left").unionByName(encoded).dropDuplicates("goods_id")
        val newCount = feature.select(s"${c}_idx").distinct().count()
        log.info(s"final distinct count: ${newCount}")
      } else {
        val newKV = makeIndex(feature, c, 0)
        val newModelPath = s"${param.rootPath}/output/TAG=${param.modelTag}/fmap/${param.featureDate}/${c}"
        newKV.saveCompressedParquet(newModelPath)
        log.info(s"new model saved to ${newModelPath}")

        log.info(s"join new KV to fit data...")
        feature = feature.withColumn(c, trim(col(c))).join(newKV, Seq(c), "left")
        val count = feature.select(s"${c}_idx").distinct().count()
        log.info(s"final distinct count: ${count}")
      }
      val cache = s"${param.rootPath}/output/TAG=${param.featureTag}/samples/tmp/${c}/${param.featureDate}"
      feature.drop(c).
        withColumnRenamed(s"${c}_idx", c).
        na.
        fill(-1,Seq(c)).
        write.
        mode("overwrite").
        parquet(cache)
      feature = spark.read.parquet(cache)
    })

    feature.show(10, false)

    log.info(s"merging encoded ckp...")
    var mergeCols = List[String]()
    val dropCols = param.dropCols.split(",")
    feature.columns.foreach(c => {
      if (!dropCols.contains(c)) {
        mergeCols = mergeCols :+ c
      }
    })

    log.info("merge columns into features...")
    val result = feature.withColumn("features",
      concat_ws(",", mergeCols.map(c => col(c).cast(LongType)):_*))

    result.show(10, false)
    log.info("saving...")
    val encodePath = s"${param.rootPath}/output/TAG=${param.featureTag}/samples/${param.featureDate}"
    result.saveCompressedParquet(encodePath)
    log.info(s"data saved to ${encodePath}")

    ToolsHdfs.delHdfsFile(log, s"${param.rootPath}/output/TAG=${param.featureTag}/samples/tmp")

  }

  def makeIndex(featureFit: DataFrame, column: String, minIndex: Long): DataFrame = {
    val indexByCount = featureFit.select(column).
      withColumn(column, trim(col(column))).
      na.drop().
      filter(s"length(${column}) > 0 ").
      groupBy(column).count().
      orderBy(desc("count")).
      rdd.map{r =>
      r.getAs[String](column).trim
    }.zipWithIndex()
    log.info(indexByCount.collect().toList.take(10))

    val rowRdd = indexByCount.map(a => Row(a._1, a._2 + minIndex))
    val schema = StructType(
      Array(
        StructField(column, StringType, true),
        StructField(s"${column}_idx", LongType, true)
      )
    )
    var df = spark.createDataFrame(rowRdd, schema)
    log.info(s"key count: ${df.count()}")
    df = df.dropDuplicates(column)
    log.info(s"unique key count: ${df.count()}")
    df
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
