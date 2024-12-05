# COVID-19 Classifier

This is a COVID-19 classifier that uses a PyTorch trained model to classify X-ray images into two categories: "infected" and "normal." The classifier is built as a Streamlit web app, allowing users to upload an X-ray image and receive a classification result.

## Usage
1. Clone the repository to your local machine.
```bash
git clone https://github.com/mirabdullahyaser/covid-classifier.git
cd covid-classifier
```
2. Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

3. Ensure you have a trained model file (replace 'trained_model.pt' with the actual path to your model file) that is compatible with this classifier.

4. Run the streamlit app.
```bash
streamlit run inference.py
```

5. Access the app by opening a web browser and navigating to the provided URL.

6. Upload an X-ray image (supported formats: jpg, png, jpeg).

7. Click the "Classify" button to perform predictions.

8. If you want to deploy it on cloud, just click on the deploy button on top right side of your streamlit app and choose Streamlit Community Cloud.
You will require to make your account on Streamlit Cloud for deployment.