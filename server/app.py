from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<body style='background-color: black;'><h1 style='color: white;'>Hello, World!</h1></body>"

def start_server():
    app.run()
