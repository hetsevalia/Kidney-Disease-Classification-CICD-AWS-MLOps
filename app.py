from flask import Flask, request, jsonify, render_template
import os
import traceback
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    try:
        os.system("python main.py")
        # os.system("dvc repro")
        return "Training done successfully!"
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        logger.info("Received prediction request")
        decodeImage(image, clApp.filename)
        logger.info("Image decoded successfully")
        result = clApp.classifier.predict()
        logger.info("Prediction completed successfully")
        return jsonify(result)
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in predictRoute: {str(e)}")
        return jsonify({
            "error": "Model not found",
            "message": "Please train the model first by clicking the 'Train Model' button.",
            "details": str(e)
        }), 404
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in predictRoute: {str(e)}\n{error_traceback}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "traceback": error_traceback
        }), 500


@app.route("/model-status", methods=['GET'])
@cross_origin()
def modelStatus():
    """Check if model exists"""
    possible_paths = [
        "artifacts/training/model.pth",
        "model/model.pth",
        "artifacts/model.pth"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return jsonify({
                "status": "ready",
                "model_path": path
            })
    
    return jsonify({
        "status": "not_found",
        "message": "Model needs to be trained"
    })


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)  # for AWS