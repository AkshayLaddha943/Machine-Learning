# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:59:20 2020

@author: Admin
"""

from flask import Flask, redirect, url_for
app = Flask(__name__)

@app.route("/")
def hello():
 return "Hello Akshay!"

@app.route("/user")
def hello_user():
 return "Akshay: Hello!"

@app.route("/admin")
def admin():
    return redirect(url_for("hello_user"))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)