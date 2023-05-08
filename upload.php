
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="//stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css" integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">
    <style>
      .bd-placeholder-img {
        font-size: 1.125rem;
        text-anchor: middle;
      }

      @media (min-width: 768px) {
        .bd-placeholder-img-lg {
          font-size: 3.5rem;
        }
      }
    </style>

    <title>Detection result</title>
  </head>
  <style type="text/css">
    h1 {text-align: center; font-size: smaller; color: rgb(179, 179, 179);}
    h2 {text-align: left}
    h3 {text-align: center; font-size: larger;}
		</style>
  <body>
  <br>
    <a href="http://127.0.0.1:8001/">	&nbsp Home	&nbsp</a>
    <a href="http://127.0.0.1:8001/autotrain/">	&nbsp Automatic	&nbsp</a>
    <a href="http://127.0.0.1:8001/manualtrain/">	&nbsp Manual	&nbsp</a>
    <a href="http://www.ihpc.se.ritsumei.ac.jp/iot/auto-test/">	&nbsp Detection	&nbsp</a>
    <form class="form-signin" method=post enctype=multipart/form-data></form>

<?php
// Path: upload.php
$allowedExts = array("gif", "jpeg", "jpg", "png","heic");
$temp = explode(".", $_FILES["file"]["name"]);

//echo $_FILES["file"]["size"];
$extension = end($temp);     // ファイルの拡張子を取得する
if ((($_FILES["file"]["type"] == "image/gif")
|| ($_FILES["file"]["type"] == "image/jpeg")
|| ($_FILES["file"]["type"] == "image/heic")
|| ($_FILES["file"]["type"] == "image/jpg")
|| ($_FILES["file"]["type"] == "image/pjpeg")
|| ($_FILES["file"]["type"] == "image/x-png")
|| ($_FILES["file"]["type"] == "image/png"))
&& ($_FILES["file"]["size"] < 20004800)   //200 kb
&& in_array($extension, $allowedExts))
{
    if ($_FILES["file"]["error"] > 0)
    {
        echo "错误：: " . $_FILES["file"]["error"] . "<br>";
    }
    else
    {
       // echo "アップロードファイル名: " . $_FILES["file"]["name"] . "<br>";
       // echo "ファイルタイプ: " . $_FILES["file"]["type"] . "<br>";
       // echo "ファイルサイズ: " . ($_FILES["file"]["size"] / 1024) . " kB<br>";
       // echo "ファイルの一時保存場所: " . $_FILES["file"]["tmp_name"] . "<br>";
        // 現在のディレクトリに upload ディレクトリが存在するかどうかを判断します
        // upload ディレクトリがない場合は、作成する必要があります。upload ディレクトリの権限は 777 に設定されています
        //if (file_exists("static/upload/" . $_FILES["file"]["name"]))
        //{
       //     echo $_FILES["file"]["name"] . " ファイルはすでに存在しています。 ";
       // }
       // else
        {
            // upload ディレクトリにそのファイルが存在しない場合、ファイルを upload ディレクトリにアップロードします
            $src = "static/img_out/";
            move_uploaded_file($_FILES["file"]["tmp_name"], "static/upload/" . $_FILES["file"]["name"]);
            //echo "ファイルが保存されている場所: " . "static/upload/" . $_FILES["file"]["name"]. "<br>";
            //echo "テスト結果: " . "static/img_out/" . $_FILES["file"]["name"]. "<br>";
            usleep(300000);
            echo "<img  style=\"text-align: center\" src=".$src.$_FILES["file"]["name"].">"."<br>";
        }
    }
}
else
{
    echo "無効なファイル形式";
}
?>
    <form method="POST">
      <h3><a href="http://www.ihpc.se.ritsumei.ac.jp/iot/auto-test/">Back</a></h3>
      </form>
      <h1><br>
          &copy; 2023 by Meng Lab. All rights reserved.</h1>
  </body>
</html>
