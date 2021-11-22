# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:40:29 2020

@author: Admin
"""

from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def first():
    if request.method == "POST":
        user= request.form["nm"]
        return redirect(url_for("user", usr=user))
    else:
        return render_template("login.html")

@app.route("/<usr>")    
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)