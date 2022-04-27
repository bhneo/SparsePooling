package com.cxyx.common.alchemist.tools

import com.github.nscala_time.time.Imports._
import com.google.common.base.Strings
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.joda.time.format.DateTimeFormatter

import java.io.IOException
import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

/**
  * Created by didi on 2020/10/13.
  */
object ToolsHdfs {

  protected def getFileSystem: FileSystem = FileSystem.get(new Configuration())

  def readRangeParquet(spark:SparkSession,log:Logger, path: String, beginDate: String, endDate: String, withDate:Boolean = false, timeFormatterPath:DateTimeFormatter = DateTimeFormat.forPattern("/yyyy/MM/dd")): DataFrame = {
    val timeFormatter = DateTimeFormat.forPattern("yyyyMMdd")
    val begin = timeFormatter.parseDateTime(beginDate)
    val end = timeFormatter.parseDateTime(endDate)

    @tailrec
    def loop(b: DateTime, e: DateTime, buf: ListBuffer[String]): List[String] = {
      if (b > e) buf.toList
      else {
        buf.append(path + b.toString(timeFormatterPath))
        loop(b + 1.days, e, buf)
      }
    }
    val paths = loop(begin, end, ListBuffer[String]())
    paths.foreach(s => log.info(s"read parquet from $s"))
    if(withDate){
      paths.map{ p =>
        val d = timeFormatterPath.parseDateTime(p.replaceAll(path, "")).toString("yyyyMMdd")
        spark.read.parquet(p).withColumn(
          "_date",
          lit(d)
        )
      }.reduce(_.union(_))
    }
    else {
      spark.read.parquet(paths: _*)
    }
  }

  def checkHdfsFileExists(log:Logger, hdfsPath: String): Boolean = {
    require(!Strings.isNullOrEmpty(hdfsPath), "input hdfs path for checking is null or empty")

    var isExists = false
    var fs: FileSystem = null

    try {
      fs = getFileSystem
      isExists = fs.exists(new Path(hdfsPath))
      log.info(s"checking HDFS (isExists=$isExists) path: $hdfsPath")
    } catch {
      case ex: IOException => throw new RuntimeException("check HDFS file faild! ", ex)
    }
    finally {
      if (fs != null) fs.close()
    }

    isExists
  }

  def getRangeParquet(spark:SparkSession,log:Logger, path: String, beginDate: String, endDate: String): Seq[String] = {
    val timeFormatter = DateTimeFormat.forPattern("yyyyMMdd")
    val begin = timeFormatter.parseDateTime(beginDate)
    val end = timeFormatter.parseDateTime(endDate)
    val timeFormatterPath = DateTimeFormat.forPattern("/yyyy/MM/dd")
    @tailrec
    def loop(b: DateTime, e: DateTime, buf: ListBuffer[String]): List[String] = {
      if (b > e) buf.toList
      else {
        buf.append(path + b.toString(timeFormatterPath))
        loop(b + 1.days, e, buf)
      }
    }
    val paths = loop(begin, end, ListBuffer[String]())

    paths
  }

  def delHdfsFile(log:Logger, hdfsPath: String): Unit = {
    require(!Strings.isNullOrEmpty(hdfsPath), "input hdfs path for checking is null or empty")

    var fs: FileSystem = null

    try {
      fs = getFileSystem
      if(fs.exists(new Path(hdfsPath))) {
        fs.delete(new Path(hdfsPath), true)
        log.info(s"hdfs path deleted: ${hdfsPath}")
      } else {
        log.info(s"path not exists: $hdfsPath")
      }
    } catch {
      case ex: IOException => throw new RuntimeException("get HDFS file faild! ", ex)
    }
    finally {
      if (fs != null) fs.close()
    }
  }

  def getHdfsFile(log:Logger, hdfsPath: String, localFile:String) = {
    require(!Strings.isNullOrEmpty(hdfsPath), "input hdfs path for checking is null or empty")

    var fs: FileSystem = null

    try {
      fs = getFileSystem
      fs.copyToLocalFile(new Path(hdfsPath),new Path(localFile))

      log.info(s"copy HDFS from : ${hdfsPath} to : ${localFile}")
    } catch {
      case ex: IOException => throw new RuntimeException("get HDFS file faild! ", ex)
    }
    finally {
      if (fs != null) fs.close()
    }
  }

  def putHdfsFile(log:Logger, hdfsPath: String, localFile:String) = {
    require(!Strings.isNullOrEmpty(hdfsPath), "input hdfs path for checking is null or empty")

    var fs: FileSystem = null

    try {
      fs = getFileSystem
      fs.copyFromLocalFile(new Path(localFile),new Path(hdfsPath))

      log.info(s"copy to HDFS from : ${localFile} to : ${hdfsPath}")
    } catch {
      case ex: IOException => throw new RuntimeException("put HDFS file faild! ", ex)
    }
    finally {
      if (fs != null) fs.close()
    }
  }

}
