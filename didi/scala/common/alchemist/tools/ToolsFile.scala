package com.cxyx.common.alchemist.tools

import java.io.{File, PrintWriter}

import org.apache.log4j.Logger

import scala.io.Source

/**
  * Created by didi on 2020/10/14.
  */
object ToolsFile {

  def checkFileExist(path:String):Boolean = {
    val file=new File(path)
    file.exists()
  }

  def delFile(log:Logger,path:String) = {
    if(checkFileExist(path)) {
      val file = new File(path)
      file.delete()
      log.info(s"[delete file]:$path")
    }
  }

  def delFileParent(log:Logger,path:String) = {
    val arr = path.split("/")
    var dir = ""
    for(i<-0 until arr.size-1) {
      dir = dir + arr(i) + "/"
    }

    deleteDir(new File(dir))
    log.info(s"[delete dir] $dir")
  }

  def getFileParentDir(path:String):String = {
    val arr = path.split("/")
    var dir = ""
    for(i<-0 until arr.size-1) {
      dir = dir + arr(i) + "/"
    }
    dir
  }

  def deleteDir(dir:File): Unit = {
    if(dir.exists()) {
      val files = dir.listFiles()
      files.foreach(f => {
        if (f.isDirectory) {
          deleteDir(f)
        } else {
          f.delete()
        }
      })
      dir.delete()
    }
  }

  def mkDir(path:String) = {
    if(!checkFileExist(path)) {
      val dir = new File(path)
      dir.mkdir()
    }
  }

  def mkDirs(path:String) = {
    if(!checkFileExist(path)) {
      val dir = new File(path)
      dir.mkdirs()
    }
  }

  def writeToFile(log:Logger, str:String,path:String) = {

    val arr = path.split("/")
    var dir = ""
    for(i<-0 until arr.size-1) {
      dir = dir + arr(i) + "/"
    }
    mkDirs(dir)

    val file = new PrintWriter(path)
    file.println(str)
    file.close()
    log.info(s"write to : $path")
  }


  def readFileSeq(path:String):Seq[String] = {
    Source.fromFile(new File(path)).getLines().toSeq
  }

}
