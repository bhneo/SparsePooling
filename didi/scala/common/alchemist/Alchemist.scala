package com.cxyx.common.alchemist


import com.alibaba.fastjson.JSON
import com.cxyx.common.lib
import com.microsoft.ml.spark.lightgbm.LightGBMClassificationModel
import com.microsoft.ml.spark.lightgbm.LightGBMRankerModel
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import scopt.OptionParser

/**
  * Created by zhaolei on 2021/04/26.
  */
object Alchemist extends lib.MainSparkApp with Constant {

  override val defaultParams: Config = Config()

  override def run(): Unit = {

    val modelConfig = JSON.parseObject(param.config)

    val (trainDF, testDF, featureCols) = DataUtils.initData(
      spark,
      param.data_root,
      param.train_suffix,
      param.test_suffix,
      param.fmap_hdfs,
      param.model_local,
      modelConfig,
      param.add_category,
      log)

    ModelUtils.makeModel(trainDF,
      modelConfig,
      featureCols,
      param.init_model_hdfs,
      param.model_local,
      param.model_hdfs,
      param.fmap_hdfs,
      param.retrain,
      log)

    val model = modelConfig.getString(MODEL_TYPE) match {

      case MODEL_TYPE_XGB =>
        val modelPath = param.model_hdfs
        println(s"load model from ${modelPath}")
        XGBoostClassificationModel.load(modelPath)

      case MODEL_TYPE_LGB =>
        val modelPath = s"${param.model_hdfs}/binary/complexParams/lightGBMBooster"
        println(s"load model from ${modelPath}")
        LightGBMClassificationModel.loadNativeModelFromFile(modelPath)

      case MODEL_TYPE_LGB_RANKER =>
        val modelPath = s"${param.model_hdfs}/binary/complexParams/lightGBMBooster"
        println(s"load model from ${modelPath}")
        LightGBMRankerModel.loadNativeModelFromFile(modelPath)

      case MODEL_TYPE_LR =>
        val modelPath = s"${param.model_hdfs}/1"
        println(s"load model from ${modelPath}")
        LogisticRegressionModel.load(modelPath)

      case _ =>
        throw new RuntimeException(s"model not support!")
    }

    if(param.explain) {
      EvaluateUtils.explainModel(model,
        param.fmap_hdfs,
        param.model_local,
        modelConfig,
        param.add_category,
        log)
    }

    EvaluateUtils.evaluate(spark,
      model,
      trainDF,
      testDF,
      s"${param.predict_root}",
      param.train_suffix,
      param.test_suffix,
      param.repredict,
      log,
      modelConfig)
  }

  override def getOptParser: OptionParser[Config] = new OptionParser[Config](" ") {
    opt[String]("data_root").required().text("data_root").
      action((x, c) => c.copy(data_root = x))
    opt[String]("predict_root").required().text("predict_root").
      action((x, c) => c.copy(predict_root = x))
    opt[String]("train_suffix").text("train_suffix").
      action((x, c) => c.copy(train_suffix = x))
    opt[String]("test_suffix").required().text("test_suffix").
      action((x, c) => c.copy(test_suffix = x))
    opt[Boolean]("add_category").optional().text("add_category").
      action((x, c) => c.copy(add_category = x))
    opt[Boolean]("retrain").required().text("retrain").
      action((x, c) => c.copy(retrain = x))
    opt[Boolean]("repredict").required().text("repredict").
      action((x, c) => c.copy(repredict = x))
    opt[Boolean]("explain").required().text("explain").
      action((x, c) => c.copy(explain = x))
    opt[String]("config").required().text("config").
      action((x, c) => c.copy(config = x))
    opt[String]("model_tag").required().text("model_tag").
      action((x, c) => c.copy(model_tag = x))
    opt[String]("init_model_hdfs").optional().text("init_model_hdfs").
      action((x, c) => c.copy(init_model_hdfs = x))
    opt[String]("model_local").required().text("model_local").
      action((x, c) => c.copy(model_local = x))
    opt[String]("model_hdfs").required().text("model_hdfs").
      action((x, c) => c.copy(model_hdfs = x))
    opt[String]("fmap_hdfs").required().text("fmap_hdfs").
      action((x, c) => c.copy(fmap_hdfs = x))
  }

  case class Config(
                     data_root: String = null,
                     predict_root: String = null,
                     train_suffix: String = null,
                     test_suffix: String = null,
                     add_category: Boolean = true,
                     retrain: Boolean = false,
                     repredict: Boolean = false,
                     explain: Boolean = true,
                     config: String = null,
                     model_tag: String = null,
                     init_model_hdfs: String = null,
                     model_local: String = null,
                     model_hdfs: String = null,
                     fmap_hdfs: String = null
                   )
}
