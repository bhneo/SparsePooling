package com.cxyx.common.lightgbm

import java.io.PrintWriter

import com.alibaba.fastjson.JSON
import com.cxyx.common.lib

import com.cxyx.common.lib.CompressDataFrameFunctionsCx._
import com.didichuxing.dm.common.tools.log.IColorText
import com.microsoft.ml.spark.lightgbm.{LightGBMClassificationModel, LightGBMClassifier}

import jodd.util.collection.IntArrayList
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.count
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import scopt.OptionParser
import com.cxyx.common.alchemist.tools.{ToolsFile, ToolsHdfs}

import scala.io.Source

/**
  * Created by didi on 2020/10/13.
  */
object LGBTrainer extends lib.MainSparkApp with IColorText {

  override val defaultParams = Config()

  var paramsConfig = ParamsConfig()

  override def run(): Unit = {
    getFmap
    getConfig
    getTrain
  }

  def getFmap = {
    log.info(s"get fmap: ${param.fmap_hdfs}")
    ToolsHdfs.getHdfsFile(log, param.fmap_hdfs, s"${param.fmap}/data")

    ToolsFile.mkDir(param.path_local_model)
  }

  def getConfig = {

    log.info(s"parsing : ${param.config}")
    val json = JSON.parseObject(param.config)
    val keyset = json.keySet()

    log.info(s"keyset : ${keyset.toArray().mkString(",")}")

    if (keyset.contains("nFold")) paramsConfig.nFold = json.getIntValue("nFold")
    if (keyset.contains("learningRate")) paramsConfig.learningRate = json.getDoubleValue("learningRate")
    if (keyset.contains("maxDepth")) paramsConfig.maxDepth = json.getIntValue("maxDepth")
    if (keyset.contains("trainingMetric")) paramsConfig.trainingMetric = json.getBooleanValue("trainingMetric")
    if (keyset.contains("barrierMode")) paramsConfig.barrierMode = json.getBooleanValue("barrierMode")
    if (keyset.contains("leaves")) paramsConfig.leaves = json.getIntValue("leaves")
    if (keyset.contains("iterations")) paramsConfig.iterations = json.getIntValue("iterations")
    if (keyset.contains("featureFraction")) paramsConfig.featureFraction = json.getDoubleValue("featureFraction")
    if (keyset.contains("baggingFraction")) paramsConfig.baggingFraction = json.getDoubleValue("baggingFraction")
    if (keyset.contains("baggingFreq")) paramsConfig.baggingFreq = json.getIntValue("baggingFreq")
    if (keyset.contains("maxBin")) paramsConfig.maxBin = json.getIntValue("maxBin")
    if (keyset.contains("categoricalSlotNames")) paramsConfig.categoricalSlotNames = json.getString("categoricalSlotNames")
    if (keyset.contains("isUnbalance")) paramsConfig.isUnbalance = json.getBooleanValue("isUnbalance")
    if (keyset.contains("minDataInLeaf")) paramsConfig.minDataInLeaf = json.getIntValue("minDataInLeaf")
    if (keyset.contains("minSumHessionInLeaf")) paramsConfig.minSumHessionInLeaf = json.getDoubleValue("minSumHessionInLeaf")
    if (keyset.contains("lambda1")) paramsConfig.lambda1 = json.getDoubleValue("lambda1")
    if (keyset.contains("lambda2")) paramsConfig.lambda2 = json.getDoubleValue("lambda2")
    if (keyset.contains("weightCol")) paramsConfig.weightCol = json.getString("weightCol")
    if (keyset.contains("workers")) paramsConfig.workers = json.getIntValue("workers")

    log.info(
      s"""
         |nFold : ${paramsConfig.nFold}
         |learningRate : ${paramsConfig.learningRate}
         |maxDepth : ${paramsConfig.maxDepth}
         |trainingMetric : ${paramsConfig.trainingMetric}
         |barrierMode : ${paramsConfig.barrierMode}
         |leaves : ${paramsConfig.leaves}
         |iterations : ${paramsConfig.iterations}
         |featureFraction : ${paramsConfig.featureFraction}
         |baggingFraction : ${paramsConfig.baggingFraction}
         |baggingFreq : ${paramsConfig.baggingFreq}
         |maxBin : ${paramsConfig.maxBin}
         |categoricalSlotNames : ${paramsConfig.categoricalSlotNames}
         |isUnbalance : ${paramsConfig.isUnbalance}
         |minSumHessionInLeaf : ${paramsConfig.minSumHessionInLeaf}
         |lambda1 : ${paramsConfig.lambda1}
         |lambda2 : ${paramsConfig.lambda2}
         |weightCol : ${paramsConfig.weightCol}
         |workers : ${paramsConfig.workers}
      """.stripMargin)
  }

  def getTrain = {
    val (trainDF, testDF) = prepare
    val model = train(trainDF)
    evaluate(model, trainDF, testDF)
  }

  def train(trainDF: DataFrame): LightGBMClassificationModel = {
    val lgboost = new LightGBMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setObjective("binary")
      .setBoostingType("gbdt")
      .setMetric("auc")
      .setIsProvideTrainingMetric(paramsConfig.trainingMetric)
      .setUseBarrierExecutionMode(paramsConfig.barrierMode)
      .setIsUnbalance(paramsConfig.isUnbalance)
      .setMaxDepth(paramsConfig.maxDepth)
      .setNumLeaves(paramsConfig.leaves)
      .setNumIterations(paramsConfig.iterations)
      .setLearningRate(paramsConfig.learningRate)
      .setLambdaL1(paramsConfig.lambda1)
      .setLambdaL2(paramsConfig.lambda2)
      .setFeatureFraction(paramsConfig.featureFraction)
      .setBaggingFraction(paramsConfig.baggingFraction)
      .setBaggingFreq(paramsConfig.baggingFreq)
      .setMaxBin(paramsConfig.maxBin)
      .setMinDataInLeaf(paramsConfig.minDataInLeaf)
      .setMinSumHessianInLeaf(paramsConfig.minSumHessionInLeaf)
      .setVerbosity(2)

    if (paramsConfig.categoricalSlotNames != "-") {
      val categoricalSlotIndexes = new IntArrayList()
      for(i <- 0 to paramsConfig.categoricalSlotNames.split(",").length) {
        categoricalSlotIndexes.add(i)
      }
      lgboost.setCategoricalSlotIndexes(categoricalSlotIndexes.toArray)
    }

    if (paramsConfig.weightCol != null && paramsConfig.weightCol != "-" && paramsConfig.weightCol.length > 0){
      lgboost.setWeightCol(paramsConfig.weightCol)
      log.info(s"lgboost.getWeightCol: ${lgboost.getWeightCol}")
    }

    log.info(s"xgboost params as follow: \n ${lgboost.explainParams()}")
    val model = lgboost.fit(trainDF)

    /******************  model写hdfs  ********************/
    model.write.overwrite().save(param.output_model)

    /******************  model写本地  ********************/
    model.saveNativeModel(s"${param.path_upload_local_model}/local_lgb_model",true)
    ToolsHdfs.getHdfsFile(
      log,
      s"${param.path_upload_local_model}/local_lgb_model",
      s"${param.path_local_model}/local_lgb_model"
    )

    /******************  fscore  ********************/
    for( x <- List("split","gain") ){
      val pathFeatureScore = s"${param.path_local_model}/fscore_${x}"
      val names = Source.fromFile(s"${param.fmap}/data").getLines().toSeq
        .map(x=>{
          x.split("\\t")(1)
        }).toArray
      var cols:Array[String] = null
      if(paramsConfig.categoricalSlotNames != "-"){
        cols = Array.concat(paramsConfig.categoricalSlotNames.split(","), names)
      } else {
        cols = names
      }
      val fvs = cols.zip(model.getFeatureImportances(s"${x}")).toMap
                  .toSeq.sortBy(_._2).reverse.map(x=>s"${x._1}\t${x._2}").mkString("\n")

      val file = new PrintWriter(pathFeatureScore)
      file.println(fvs)
      file.close()
      log.info(s"save split feature score to : $pathFeatureScore")

      log.info(Yellow("[DEBUG] ") + "--------------- topK feature score")
      Source.fromFile(pathFeatureScore, "utf-8").getLines().toSeq
        .slice(0, 100).zipWithIndex
        .foreach(x => println(s"${x._2}\t${x._1}"))
    }

    val numFeatures = model.numFeatures
    log.info(s"save model (numFeatures: $numFeatures) HDFS: ${param.output_model} local: ${param.path_local_model}")

    model
  }

  def evaluate(model: LightGBMClassificationModel,
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

    // 输出预测结果的十分位数
    println("----------- 整个训练集合的分布 （train + test） -----------")
    Utils.showProbPercentiles(predictTrainDF.union(predictTestDF))

  }


  def getDataSet:DataFrame = {
    import spark.implicits._

    val cache = param.path_train_pre + "/cache"
    if(ToolsHdfs.checkHdfsFileExists(log, cache)) {
      log.info(s"${cache} exists, read it directly!")
      spark.read.parquet(cache)
    } else {
      log.info(s"${cache} not exists, read from original files!")
      val originData = ToolsHdfs.readRangeParquet(spark,log,param.path_train_pre,param.date_start,param.date_end)

      val pos = originData.filter($"label"===1.0).filter($"rand"<=param.pos_ratio).cache()
      log.info(s"pos count : ${pos.count()}")

      val neg = originData.filter($"label"===0.0).filter($"rand"<=param.neg_ratio).cache()
      log.info(s"neg count : ${neg.count()}")

      val data= pos.union(neg)
      data.saveCompressedParquet(cache)
      spark.read.parquet(cache)
    }
  }

  def prepare: (DataFrame, DataFrame) = {

    import spark.implicits._

    val originData = getDataSet
    originData.cache()
    log.info(s"originData count : ${originData.count()}")

    var df:DataFrame = null
    // 除去哪些特征
    if(param.feature_exclude!=null && !param.feature_exclude.equals("")) {
      log.info(s"remove features : ${param.feature_exclude}")
      val map = Source.fromFile(s"${param.fmap}/data").getLines().toSeq
        .map(x=>{
          (x.split("\\t")(1),x.split("\\t")(0).toInt)
        }).toMap

      val index = param.feature_exclude.split(",").toSeq.distinct
        .map(x=>map.getOrElse(x,-1))
        .filter(_ > -1)
      log.info(s"remove feature index : ${index.mkString(",")}")
      val indexBC = spark.sparkContext.broadcast(index)

      df = originData.map(r=>{
        val features = r.getAs[Vector]("features").toArray
        indexBC.value.foreach(i=>features(i)=0.0)
        (
          r.getAs[String]("user_id"),
          r.getAs[String]("goods_id"),
          r.getAs[Double]("label"),
          new DenseVector(features)
        )
      }).toDF("user_id","goods_id","label","features")
    } else {
      df = originData
        .map(r=>{
          val features = r.getAs[Vector]("features").toArray
          (
            r.getAs[String]("user_id"),
            r.getAs[String]("goods_id"),
            r.getAs[Double]("label"),
            new DenseVector(features)
          )
        }).toDF("user_id","goods_id","label","features")
    }

    if (paramsConfig.weightCol != null && paramsConfig.weightCol != "-" && paramsConfig.weightCol.length > 0){
      df = df.join(originData.select("user_id","goods_id","time","weight"),
        Seq("user_id","goods_id","time"), "left")
    }

    df.cache()
    log.info(s"all set count : ${df.count()}")
    df.groupBy($"label").agg(count("user_id").as("cnt")).show()

    val inputDF = mergeCategoryFeatures(df)

    val (train_data, test_data) = if (paramsConfig.nFold > 1) {
      val test_ratio = 1.0 / paramsConfig.nFold
      val arrDF = inputDF.randomSplit(Array(1 - test_ratio, test_ratio), seed = 11L)
      (arrDF(0), arrDF(1))
    } else {
      val arrDF = inputDF.randomSplit(Array(1 - 0.3, 0.3), seed = 11L)
      (arrDF(0), arrDF(1))
    }

    val train = train_data.repartition(paramsConfig.workers).cache()
    val test = test_data.cache()

    val n_train = train.count()
    if (n_train < 1)
      log.error("Training data is empty!")

    val n_test = test.count()

    val n_class = train.dropDuplicates("label").count().toInt
    val n_feature = train.first().getAs[Vector]("features").size
    log.info(s"n(train)=${n_train} n(test)=${n_test} n(class)=${n_class} n(feature)=${n_feature}")

    (train, test)
  }

  def mergeCategoryFeatures(df: DataFrame): DataFrame = {
    // 离散列
    val cateCols = paramsConfig.categoricalSlotNames
    log.info(s"categorical columns: ${cateCols}")
    // 原始列
    val conCols = Array("features")
    // feature列
    var vecCols:Array[String] = null
    if(cateCols != "-"){
      vecCols = cateCols.split(",") ++ conCols
    } else {
      vecCols = conCols
    }

    var inputDF = df.select("label", vecCols: _*)
    import spark.implicits._
    if(cateCols != "-"){
      log.info(s"cast categorical columns to double...")
      cateCols.split(",").foreach(col => {
        inputDF = inputDF.withColumn(col, $"$col".cast(DoubleType))
      })
    }

    inputDF = inputDF.withColumn("label", $"label".cast(IntegerType))

    val assembler = new VectorAssembler().setInputCols(vecCols).setOutputCol("merged_features")
    inputDF = assembler.transform(inputDF).drop("features").withColumnRenamed("merged_features", "features")
    log.info(s"assembled dataframe:")
    inputDF.show(10)
    inputDF
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

  case class ParamsConfig(
                           var nFold: Int = 5,
                           var learningRate: Double = 0.1,
                           var trainingMetric: Boolean = false,
                           var barrierMode: Boolean = false,
                           var maxDepth: Int = -1,
                           var leaves: Int = 31,
                           var iterations: Int = 100,
                           var featureFraction: Double = 1.0,
                           var baggingFraction: Double = 1.0,
                           var baggingFreq: Int = 0,
                           var maxBin: Int = 255,
                           var categoricalSlotNames: String = "",
                           var isUnbalance: Boolean = false,
                           var minDataInLeaf: Int = 20,
                           var minSumHessionInLeaf: Double = 1e-3,
                           var lambda1: Double = 0.0,
                           var lambda2: Double = 0.0,
                           var weightCol:String = "",
                           var workers: Int = 500
                         )

  override def getOptParser: OptionParser[Config] = new OptionParser[Config](" ") {
    head("feature")
    opt[Double]("pos_ratio").text("pos_ratio").
      action((x, c) => c.copy(pos_ratio = x))
    opt[Double]("neg_ratio").text("neg_ratio").
      action((x, c) => c.copy(neg_ratio = x))
    opt[String]("feature_exclude").text("feature_exclude").
      action((x, c) => c.copy(feature_exclude = x))
    opt[String]("date_start").text("date_start").
      action((x, c) => c.copy(date_start = x))
    opt[String]("date_end").required().text("date_end").
      action((x, c) => c.copy(date_end = x))
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
                     pos_ratio: Double = 1.0,
                     neg_ratio: Double = 1.0,
                     feature_exclude: String = null,
                     date_start: String = null,
                     date_end: String = null,
                     path_train_pre: String = null,
                     config: String = null,
                     fmap_hdfs: String = null,
                     fmap: String = null,
                     path_local_model: String = null,
                     path_upload_local_model: String = null,
                     output_model: String = null
                   )
}
