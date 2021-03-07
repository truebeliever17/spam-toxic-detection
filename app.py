from flask import Flask, request, render_template
from models import load_model, load_w2v, load_tokenizer, SpamToxicDetector

app = Flask(__name__, template_folder='templates')


idx2label = {0: 'normal', 1: 'toxic', 2: 'spam'}
num_classes = len(idx2label)
word2vec = load_w2v()
tokenizer = load_tokenizer()
model = load_model('./model.pt', word2vec.vector_size, num_classes)

classifier = SpamToxicDetector(model, tokenizer, word2vec, idx2label)


@app.route("/", methods=['GET', 'POST'])
def index():
    label = None
    if request.method == 'POST':
        message = request.form.get('message')
        label = classifier.make_prediction(message)

    return render_template('index.html', label=label)


@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        text = request.form['message'].strip()
        if len(text) > 0:
            return {'label': classifier.make_prediction(text)}
        else:
            return 'Message is empty'
    except KeyError:
        return "Field 'message' is not in the request body!"