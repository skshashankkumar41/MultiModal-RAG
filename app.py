from src.pdf_handler.pdf_handler import PdfHandler
from src.llama_handler.llama_handler import LlamaHandler
from flask import Flask, request, jsonify, render_template
import logging.config

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',handlers=[logging.FileHandler("logging.log", mode='w'),logging.StreamHandler()])
app.logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# instance of LlamaHanlder
llama_handler = LlamaHandler(llm=None, mm_llm='llava')
# llama_handler = LlamaHandler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/persist', methods=['POST'])
def pipeline_v1_persist():
    try:
        pdf_handler = PdfHandler()
        pdf_handler.process_pdfs()
        llama_handler.pipeline_v1_persist()
        return jsonify({'message': 'Pipeline v1 persisted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/persist_v3', methods=['POST'])
def pipeline_v3_persist():
    try:
        pdf_handler = PdfHandler()
        pdf_handler.process_pdfs()
        llama_handler.pipeline_v3_persist()
        return jsonify({'message': 'Pipeline v3 persisted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/answer_v3', methods=['POST'])
def answer_v3():
    try:
        question = request.json['question']
        response = llama_handler.multi_modal_answer_engine_v3(question)
        return jsonify({'answer': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/answer', methods=['POST'])
def answer():
    try:
        question = request.json['question']
        response = llama_handler.multi_modal_answer_engine(question)
        return jsonify({'answer': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=False)
    


