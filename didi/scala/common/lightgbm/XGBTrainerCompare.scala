package com.cxyx.common.lightgbm

import java.io.PrintWriter
import java.nio.file.{Files, Paths}

import com.alibaba.fastjson.JSON
import com.cxyx.common.lib
import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.didichuxing.dm.common.tools.log.IColorText
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import jodd.util.collection.IntArrayList
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import scopt.OptionParser
import com.cxyx.common.alchemist.tools.{ToolsFile, ToolsHdfs}

import scala.io.Source

/**
  * Created by didi on 2020/10/13.
  */
object XGBTrainerCompare extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()

  var paramsConfig = ParamsConfig()

  override def run(): Unit = {
    getFMap
    getConfig
    getTrain
  }

  def getFMap = {
    log.info(s"get fmap: ${param.fmap_hdfs}")
    ToolsHdfs.getHdfsFile(log, param.fmap_hdfs, s"${param.fmap}/data")

    ToolsFile.mkDir(param.path_local_model)
  }

  def getConfig = {
    log.info(s"parsing : ${param.config}")
    val json = JSON.parseObject(param.config)
    val keyset = json.keySet()

    log.info(s"keyset : ${keyset.toArray().mkString(",")}")

    if (keyset.contains("eta")) paramsConfig.eta = json.getDoubleValue("eta")
    if (keyset.contains("maxDepth")) paramsConfig.maxDepth = json.getIntValue("maxDepth")
    if (keyset.contains("round")) paramsConfig.round = json.getIntValue("round")
    if (keyset.contains("workers")) paramsConfig.workers = json.getIntValue("workers")
    if (keyset.contains("autoScalePosRatio")) paramsConfig.autoScalePosRatio = json.getIntValue("autoScalePosRatio")
    if (keyset.contains("scale_pos_weight")) paramsConfig.scale_pos_weight = json.getDoubleValue("scale_pos_weight")
    if (keyset.contains("minChildWeight")) paramsConfig.minChildWeight = json.getIntValue("minChildWeight")
    if (keyset.contains("alpha")) paramsConfig.alpha = json.getDoubleValue("alpha")
    if (keyset.contains("lambda")) paramsConfig.lambda = json.getDoubleValue("lambda")
    if (keyset.contains("gamma")) paramsConfig.gamma = json.getDoubleValue("gamma")
    if (keyset.contains("weight_col")) paramsConfig.weight_col = json.getString("weight_col")
    if (keyset.contains("categoricalSlotNames")) paramsConfig.categoricalSlotNames = json.getString("categoricalSlotNames")


    log.info(
      s"""
         |eta : ${paramsConfig.eta}
         |maxDepth : ${paramsConfig.maxDepth}
         |round : ${paramsConfig.round}
         |workers : ${paramsConfig.workers}
         |autoScalePosRatio : ${paramsConfig.autoScalePosRatio}
         |minChildWeight : ${paramsConfig.minChildWeight}
         |alpha : ${paramsConfig.alpha}
         |lambda : ${paramsConfig.lambda}
         |gamma : ${paramsConfig.gamma}
         |weight_col : ${paramsConfig.weight_col}
         |categoricalSlotNames : ${paramsConfig.categoricalSlotNames}
      """.stripMargin)
  }

  def getTrain = {
    val (trainDF, testDF) = prepare
    val model = train(trainDF)
    evaluate(model, trainDF, testDF)
  }

  def train(trainDF: DataFrame): XGBoostClassificationModel = {
    // 自动设置worker数目
    val n_executor = spark.sparkContext.getExecutorMemoryStatus.size - 1
    val n_executor_core = spark.conf.get("spark.executor.cores").toInt
    var n_worker = n_executor * n_executor_core
    val n_raw_partition = trainDF.rdd.getNumPartitions
    if (n_worker > n_raw_partition) {
      log.warn(s"number of executors($n_worker) should not be larger than num_partition($n_raw_partition), reset it to num_partition")
      n_worker = n_raw_partition
    }
    n_worker = paramsConfig.workers
    log.info(s"n(worker)=$n_worker, n(partition)=$n_raw_partition")


    val scalePosWeight = getScalePosWeight(trainDF, paramsConfig.autoScalePosRatio, paramsConfig.scale_pos_weight)

    val xgboost = new XGBoostClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setObjective("binary:logistic")
      .setNumWorkers(n_worker)
      .setMaxDepth(paramsConfig.maxDepth)
      .setNumRound(paramsConfig.round)
      .setEta(paramsConfig.eta)
      .setMinChildWeight(paramsConfig.minChildWeight)
      .setAlpha(paramsConfig.alpha)
      .setLambda(paramsConfig.lambda)
      .setGamma(paramsConfig.gamma)
      .setEvalMetric("auc")
      .setScalePosWeight(scalePosWeight)
      .setTreeMethod("hist")
      .setTrainTestRatio(0.9)
//      .setMissing(0.0f)

    log.info(s"xgboost params as follow: \n ${xgboost.explainParams()}")
    val model = xgboost.fit(trainDF)

    /******************  model写hdfs  ********************/
    model.write.overwrite().save(param.output_model)

    /******************  model写本地  ********************/
    model.nativeBooster.saveModel(s"${param.path_local_model}/local_xgb_model")
    ToolsHdfs.putHdfsFile(
      log,
      s"${param.path_upload_local_model}/local_xgb_model",
      s"${param.path_local_model}/local_xgb_model"
    )

    saveTreeToLocalFile(s"${param.path_local_model}/local_xgb_tree", model,
      s"${param.fmap}/data")
    log.info(s"save tree: ${param.path_local_model}")

    /******************  fscore  ********************/
    val path_fscore = s"${param.path_local_model}/fscore"
    val fvs = model.nativeBooster.getFeatureScore(s"${param.fmap}/data").toMap
      .toSeq.sortBy(_._2).reverse.map(x=>s"${x._1}\t${x._2}").mkString("\n")
    val file = new PrintWriter(path_fscore)
    file.println(fvs)
    file.close()
    log.info(s"save fscore to : $path_fscore")

    log.info(Yellow("[DEBUG] ") + "--------------- topK feature score")
    Source.fromFile(path_fscore, "utf-8").getLines().toSeq
      .slice(0, 100).zipWithIndex
      .foreach(x => println(s"${x._2}\t${x._1}"))

    val numFeatures = model.numFeatures
    log.info(s"save model (numFeatures: $numFeatures) HDFS: ${param.output_model} local: ${param.path_local_model}")

    log.info(model.summary.toString())

    model
  }

  def evaluate(model: XGBoostClassificationModel,
               trainDF: DataFrame,
               testDF: DataFrame
              ) = {

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    // 为了方便复制，这些参数不要用log形式
    println("----------- Evaluation on training data -----------")
    val predictTrainDF = Utils.customPrediction(model.transform(trainDF))
    val trainAUC = evaluator.evaluate(predictTrainDF)
    println(s"AUC on train data $trainAUC")

    println("----------- Evaluation on test data -----------")
    val predictTestDF = Utils.customPrediction(model.transform(testDF))
    val testAUC = evaluator.evaluate(predictTestDF)
    println(s"AUC on test data $testAUC")
    predictTestDF.saveCompressedParquet(param.path_train_pre + "/predict/test")

    // 输出预测结果的十分位数
    println("----------- 整个训练集合的分布 （train + test） -----------")
    Utils.showProbPercentiles(predictTrainDF.union(predictTestDF))

  }

  private def saveTreeToLocalFile(filePath: String, model: XGBoostClassificationModel, fmap: String): Unit = {
    val content = model.nativeBooster.getModelDump(fmap, true).mkString("")
    try {
      Files.delete(Paths.get(filePath))
    } catch {
      case _: Exception =>
    }
    Files.write(Paths.get(filePath), content.getBytes)
  }

  def getDataSet: (DataFrame, DataFrame) = {

    val trainCache = param.path_train_pre + "/train"
    val trainData = if(ToolsHdfs.checkHdfsFileExists(log, trainCache)) {
      log.info(s"${trainCache} exists, read it directly!")
      spark.read.parquet(trainCache)
    } else {
      log.info(s"${trainCache} not exists, read from original files!")
      val originData = ToolsHdfs.readRangeParquet(spark,log,param.path_train_pre,param.train_start,param.train_end)

      originData.saveCompressedParquet(trainCache)
      spark.read.parquet(trainCache)
    }

    log.info(s"train count : ${trainData.count()}")
    trainData.groupBy("label").count().show()

    val testData = spark.read.parquet(s"${param.path_train_pre}/${param.test_date}")
    log.info(s"test count : ${testData.count()}")
    testData.groupBy("label").count().show()

    (trainData, testData)
  }

  def prepare: (DataFrame, DataFrame) = {

    import spark.implicits._

    val (trainDFOri, testDFOri) = getDataSet

    var trainDF = trainDFOri
      .map(r=>{
        val features = r.getAs[Vector]("features").toArray
        (
          r.getAs[String]("user_id"),
          r.getAs[String]("goods_id"),
          r.getAs[Double]("label"),
          new DenseVector(features)
        )
      }).toDF("user_id","goods_id","label","features")
    var testDF = testDFOri
      .map(r=>{
        val features = r.getAs[Vector]("features").toArray
        (
          r.getAs[String]("user_id"),
          r.getAs[String]("goods_id"),
          r.getAs[Double]("label"),
          new DenseVector(features)
        )
      }).toDF("user_id","goods_id","label","features")

    val n_train = trainDF.count()
    if (n_train < 1)
      log.error("Training data is empty!")

    val n_test = testDF.count()

    val n_class = trainDF.dropDuplicates("label").count().toInt
    val n_feature = trainDF.first().getAs[Vector]("features").size
    log.info(s"n(train)=${n_train} n(test)=${n_test} n(class)=${n_class} n(feature)=${n_feature}")

    trainDF = mergeCategoryFeatures(trainDF.join(trainDFOri.drop("features"),Seq("user_id","goods_id","label"),"left"))
    testDF = mergeCategoryFeatures(testDF.join(testDFOri.drop("features"),Seq("user_id","goods_id","label"),"left"))

    trainDF = trainDF.repartition(paramsConfig.workers).cache()
    testDF = testDF.cache()

    (trainDF, testDF)
  }

  def getScalePosWeight(df: DataFrame, autoScalePosRatio: Int, scalePosWeight: Double = -1.0): Double = {

    // 如果设置了scalePosWeight，直接使用指定的值
    if (scalePosWeight > 1.0) {
      log.info(Yellow("[AUTO] ") + s"直接使用指定的 scalePosWeight: ${scalePosWeight}")
      return scalePosWeight
    }

    // 看是否启用了自动计算scalePosWeight
    if (autoScalePosRatio < 1) {
      log.info(Red("[AUTO] ") + "输入的 autoScalePosRatio < 1.0，设置 scalePosWeight=1.0")
      return 1.0
    } else {
      // 按输入的ratio，根据正负例的差值，进行调整
      val neg = df.filter("label = 0.0").count()
      val pos = df.filter("label = 1.0").count()
      val rawRatio = neg.toFloat / pos.toFloat
      val scalePosWeight = if (pos < neg) {
        rawRatio / autoScalePosRatio
      } else {
        log.info("正例 > 负例，scalePosWeight 强制设置为：1.0")
        1.0
      }
      log.info(Yellow("[AUTO] ") + s"自动调整 scalePosWeight: Pos=${pos} Neg=${neg} Neg/Pos=${rawRatio} autoScalePosRatio=${autoScalePosRatio} scalePosWeight=${scalePosWeight}")

      scalePosWeight
    }
  }

  def mergeCategoryFeatures(df: DataFrame): DataFrame = {
    import spark.implicits._
    var inputDF = df.withColumn("label", $"label".cast(IntegerType))
    // 离散列
    val cateCols = paramsConfig.categoricalSlotNames
    log.info(s"categorical columns: ${cateCols}")
    // feature列
    var vecCols:Array[String] = null
    if(cateCols != "-"){
      // 原始列
      val conCols = Array("features")
      vecCols = cateCols.split(",") ++ conCols
      log.info(s"vecCols: ${vecCols}")
      inputDF = inputDF.select("label", vecCols: _*)
      if(cateCols != "-"){
        log.info(s"cast categorical columns to double...")
        cateCols.split(",").foreach(col => {
          inputDF = inputDF.withColumn(col, $"$col".cast(DoubleType))
        })
      }

      val assembler = new VectorAssembler().setInputCols(vecCols).setOutputCol("merged_features")
      inputDF = assembler.transform(inputDF).drop("features").withColumnRenamed("merged_features", "features")
      log.info(s"assembled dataframe:")
      inputDF.show(10)
    } 
    inputDF
  }

  case class ParamsConfig(
                           var eta: Double = 0.08,
                           var maxDepth: Int = 5,
                           var round: Int = 200,
                           var workers: Int = 100,
                           var autoScalePosRatio: Int = 1,
                           var scale_pos_weight: Double = 1.0,
                           var minChildWeight: Int = 20,
                           var alpha: Double = 0.1,
                           var lambda: Double = 0.1,
                           var gamma:Double = 0.1,
                           var weight_col:String = "",
                           var categoricalSlotNames: String = ""
                         )

  override def getOptParser: OptionParser[Config] = new OptionParser[Config](" ") {
    head("feature")
    opt[String]("train_start").text("train_start").
      action((x, c) => c.copy(train_start = x))
    opt[String]("train_end").required().text("train_end").
      action((x, c) => c.copy(train_end = x))
    opt[String]("test_date").required().text("test_date").
      action((x, c) => c.copy(test_date = x))
    opt[String]("path_train_pre").required().text("path_train_pre").
      action((x, c) => c.copy(path_train_pre = x))
    opt[String]("config").required().text("config").
      action((x, c) => c.copy(config = x))
    opt[String]("fmap_hdfs").required().text("fmap_hdfs").
      action((x, c) => c.copy(fmap_hdfs = x))
    opt[String]("fmap").required().text("fmap").
      action((x, c) => c.copy(fmap = x))
    opt[String]("path_local_model").required().text("path_local_model").
      action((x, c) => c.copy(path_local_model = x))
    opt[String]("path_upload_local_model").required().text("path_upload_local_model").
      action((x, c) => c.copy(path_upload_local_model = x))
    opt[String]("output_model").required().text("output_model").
      action((x, c) => c.copy(output_model = x))
  }

  case class Config(
                     train_start: String = null,
                     train_end: String = null,
                     test_date: String = null,
                     path_train_pre: String = null,
                     config: String = null,
                     fmap_hdfs: String = null,
                     fmap: String = null,
                     path_local_model: String = null,
                     path_upload_local_model: String = null,
                     output_model: String = null
                   )
}
