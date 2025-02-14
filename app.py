from flask import Flask, request, render_template, send_file
from model import generate_style_transfer
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded files
        content_file = request.files["content"]
        style_file = request.files["style"]

        # Save uploaded images
        content_path = os.path.join(UPLOAD_FOLDER, "content.jpg")
        style_path = os.path.join(UPLOAD_FOLDER, "style.jpg")
        content_file.save(content_path)
        style_file.save(style_path)

        # Generate the styled image
        styled_image = generate_style_transfer(content_path, style_path)
        
        # Save the result
        result_path = os.path.join(RESULT_FOLDER, "styled_result.jpg")
        cv2.imwrite(result_path, styled_image[0] * 255)

        return render_template("result.html", result_image=result_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
