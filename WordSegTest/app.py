#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from gevent import monkey
monkey.patch_all()
from gevent import pywsgi
from flask import Flask, Response
from wordSegTest import Segment

cur_dir = os.getcwd() or os.path.dirname(__file__)
app = Flask(__name__)

@app.route("/segment/<text>", methods = ["GET"])
def segment(text):
    seg = Segment(cur_dir + "/user_dict.txt", cur_dir + "/stopwords.txt", cur_dir + "/sensewords.txt")
    res = seg.word_cut(text)
    r = Response(response=res, status=200, mimetype="application/json")
    return r

if __name__ == "__main__":
    server = pywsgi.WSGIServer(("10.126.92.200", 8050), app)
    server.serve_forever()
