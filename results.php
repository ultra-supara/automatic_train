<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Detection results</title>
  </head>
  <style type="text/css">
    h1 {text-align: center; font-size: smaller; color: rgb(179, 179, 179);}
    h2 {text-align: left}
    h3 {text-align: center; font-size: medium;}
		</style>
  <body>
  <br>
    <a href="http://127.0.0.1:8001/">	&nbsp Home	&nbsp</a>
    <a href="http://127.0.0.1:8001/autotrain/">	&nbsp Automatic	&nbsp</a>
    <a href="http://127.0.0.1:8001/manualtrain/">	&nbsp Manual	&nbsp</a>
    <a href="http://www.ihpc.se.ritsumei.ac.jp/iot/auto-test/">	&nbsp Detection	&nbsp</a>
    <br>
    <form class="form-signin" method=post enctype=multipart/form-data></form>
    <h3><img class="" src="{{url_for('static',filename='img_out/'+image_url)}}" alt="" width="320" high=""/></h3>
    {% if txt %}
    <br>
      <h3> <font = 4pt color = "red">Detection failed, please retake the photo.</font></h3>
    {% endif %}
    <form method="POST">
      <h3><input type=submit value='Back' name="Back"></h3>
      </form>
      <h1>&copy; 2023 by Meng Lab. All rights reserved.</h1>
  </body>
</html>
