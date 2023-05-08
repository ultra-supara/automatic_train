<?php
session_start();
if(isset($_POST['submit'])){
    $name = $_FILES["image-file"]["name"];
    $ext = pathinfo($name, PATHINFO_EXTENSION);
    if($ext == 'jpg' or $ext == 'jpeg' or $ext == 'png'){
        if($ext == 'png'){
            $image = imagecreatefrompng($_FILES["image-file"]["tmp_name"]);
            $bg = imagecreatetruecolor(imagesx($image), imagesy($image));
            imagefill($bg, 0, 0, imagecolorallocate($bg, 255, 255, 255));
            imagealphablending($bg, TRUE);
            imagecopy($bg, $image, 0, 0, 0, 0, imagesx($image), imagesy($image));
            imagedestroy($image);
            $quality = 70;
            imagejpeg($bg, "page.jpg", $quality);
            imagedestroy($bg);
            header('Location: results.php');
        }
        else{
            if(move_uploaded_file($_FILES["image-file"]["tmp_name"],"page.jpg")){
                header('Location: results.php');
            }
            else{
                echo '<script>alert("Failed to upload Image.");</script>';
            }
        }
    }
    else{
        echo '<script>alert("Uploaded format does not belong to an image.Allowed formats: JPG, JPEG, PNG");</script>';
    }
}
?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="//stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css" integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">
    <title>Dish Detection</title>
  </head>
  <style type="text/css">
    h1 {text-align: center; font-size: small;}
    h2 {text-align: left}
    h3 {text-align: center; font-size: medium; color: rgb(179, 179, 179);}
  </style>
  <body>
  <br>
    <a href="http://www.ihpc.se.ritsumei.ac.jp/"><img class="" src="/static/meng.png" alt="" width="160" high="160"></a>
    <a href="http://127.0.0.1:8001/">	&nbsp Home	&nbsp</a>
    <a href="http://127.0.0.1:8001/autotrain/">	&nbsp Automatic	&nbsp</a>
    <a href="http://127.0.0.1:8001/manualtrain/">	&nbsp Manual	&nbsp</a>
    <a href="http://www.ihpc.se.ritsumei.ac.jp/iot/auto-test/">	&nbsp Detection	&nbsp</a>
    <br><br><br>
    <form action="upload.php" class="form-signin" method=post enctype=multipart/form-data>
        <h1 class="h3 mb-3 font-weight-normal">Dish Detection</h1>
        <br>
        <br>
        <input type="file" name="file" accept="image/*" class="form-control-file" id="file" multiple> 
        <br>
        <button class="btn btn-lg btn-primary btn-block" type="submit">Upload</button><br> 
        <h3><br>
          &copy; 2023 by Meng Lab. All rights reserved.</h3>
    </form>

  </body>
</html>
