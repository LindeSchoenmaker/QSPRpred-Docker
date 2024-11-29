import logging
from flask import Flask, request, render_template, jsonify, Response, send_file
from qsprpred.models import SklearnModel
import os
import io
import csv
import json
import traceback
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from rdkit import Chem
from rdkit.Chem import Draw
import base64

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the models directory
MODELS_DIR = 'models'

# define RDKit implementation
def smiles_to_image(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles) # attempt conversion to RDKit molecule
        if mol is None: # return None if not possible
            return None
        img = Draw.MolToImage(mol) # Image generation
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logging.error(f"Error generating image for SMILES {smiles}: {e}") # Log the error message if any exception occurs during the process.
        return None
    
def extract_model_info(directory):
    models_info = []
    for d in os.listdir(directory):
        meta_path = os.path.join(directory, d, f"{d}_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, 'r') as meta_file:
                meta_data = json.load(meta_file)
                state = meta_data['py/state']
                model_info = {
                    'name': state['name'],
                    'target_property_name': state['targetProperties'][0]['py/state']['name'],
                    'target_property_task': state['targetProperties'][0]['py/state']['task']['py/reduce'][1]['py/tuple'][0],
                    'feature_calculator': state['featureCalculators'][0]['py/object'].split('.')[-1],
                    'radius': state['featureCalculators'][0]['py/state']['radius'],
                    'nBits': state['featureCalculators'][0]['py/state']['nBits'],
                    'algorithm': state['alg'].split('.')[-1]
                }
                models_info.append(model_info)
                logging.info(f"Loaded model metadata: {model_info}")
    return models_info

@app.route('/')
def home():
    available_models = extract_model_info(MODELS_DIR)
    return render_template('index.html', models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Handling prediction request.")
    available_models = extract_model_info(MODELS_DIR)
    try:
        smiles_input = request.form.get('smiles')
        uploaded_file = request.files.get('file')
        model_names = request.form.getlist('model')
        file_name = request.form.get('uploaded_file_name')
        
        logging.debug(f"Received SMILES input: {smiles_input}")
        logging.debug(f"Uploaded file: {uploaded_file}")
        logging.debug(f"Selected models: {model_names}")
        logging.debug(f"Previous uploaded file name: {file_name}")
        
        if not model_names:
            logging.error("No model selected.")
            return render_template('index.html', models=available_models, error="No model selected.")
        
        smiles_list = []
        if smiles_input:
            smiles_list.extend([smile.strip() for smile in smiles_input.split(',')])
        
        if uploaded_file and uploaded_file.filename != '':
            file_name = uploaded_file.filename
            logging.debug("Processing uploaded file.")
            uploaded_df = pd.read_csv(uploaded_file)
            logging.debug(f"Uploaded file contents: {uploaded_df.head()}")
            if 'SMILES' in uploaded_df.columns:
                smiles_list.extend(uploaded_df['SMILES'].tolist())
        elif file_name:
            # Reprocess the previously uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            if os.path.exists(file_path):
                logging.debug("Reprocessing previous uploaded file.")
                uploaded_df = pd.read_csv(file_path)
                if 'SMILES' in uploaded_df.columns:
                    smiles_list.extend(uploaded_df['SMILES'].tolist())
        
        logging.debug(f"Final SMILES list: {smiles_list}")
        
        if not smiles_list:
            logging.error("No SMILES strings provided.")
            return render_template('index.html', models=available_models, error="No SMILES strings provided.")
        
        all_predictions = {}
        model_info_list = []
        for model_name in model_names:
            logging.debug(f"Processing model: {model_name}")
            model_path = os.path.join(MODELS_DIR, model_name, f"{model_name}_meta.json")
            model = SklearnModel.fromFile(model_path)
            predictions = model.predictMols(smiles_list)
            
            # Check if the model is regression or classification
            if model.task.isRegression():
                # Format regression output as numeric values
                formatted_predictions = [f"{pred[0]:.2f}" for pred in predictions] # :.2f defines decimals
            else:
                # Format classification output as Active/Inactive
                formatted_predictions = ["Active" if pred[0] == 1 else "Inactive" for pred in predictions]
            
            all_predictions[model_name] = formatted_predictions
            
            # Extract model info for report
            with open(model_path, 'r') as meta_file:
                meta_data = json.load(meta_file)
                state = meta_data['py/state']
                model_info = {
                    'name': state['name'],
                    'target_property_name': state['targetProperties'][0]['py/state']['name'],
                    'target_property_task': state['targetProperties'][0]['py/state']['task']['py/reduce'][1]['py/tuple'][0],
                    'feature_calculator': state['featureCalculators'][0]['py/object'].split('.')[-1],
                    'radius': state['featureCalculators'][0]['py/state']['radius'],
                    'nBits': state['featureCalculators'][0]['py/state']['nBits'],
                    'algorithm': state['alg'].split('.')[-1]
                }
                model_info_list.append(model_info)
        
        table_data = []
        for i, smile in enumerate(smiles_list): 
            image_data = smiles_to_image(smile)
            row = [image_data] + [all_predictions[model][i] for model in model_names]
            table_data.append(row)
            
        # Update headers
        headers = ['Structure']
        for model_name in model_names:
            model_path = os.path.join(MODELS_DIR, model_name, f"{model_name}_meta.json")               
            model = SklearnModel.fromFile(model_path)
            
            if model.task.isRegression():
                # Format regression table header
                headers.append(f'Predicted pChEMBL Value ({model_name})')
            else:
                # Format classification table header
                headers.append(f'Predicted class label ({model_name})')

        # Handle report download request
        if 'download_report' in request.form:
            report_buffer = create_report(model_info_list, headers, table_data)
            return send_file(report_buffer, as_attachment=True, download_name="prediction_report.pdf", mimetype='application/pdf')
        
        return render_template('index.html', models=available_models, headers=headers, data=table_data, smiles_input=smiles_input, model_names=model_names, file_name=file_name)
    except Exception as e:
        logging.exception("An error occurred while processing the request.")
        return render_template('index.html', models=available_models, error="An error occurred while processing the request.")

def create_report(model_info_list, headers, table_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Add title
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    elements.append(Paragraph("Prediction Report", title_style))
    elements.append(Spacer(1, 12))

    # Add model metadata
    for model_info in model_info_list:
        elements.append(Paragraph(f"Model: {model_info['name']}", styles['Heading2']))
        elements.append(Paragraph(f"Target Property Name: {model_info['target_property_name']}", styles['Normal']))
        elements.append(Paragraph(f"Target Property Task: {model_info['target_property_task']}", styles['Normal']))
        elements.append(Paragraph(f"Feature Calculator: {model_info['feature_calculator']}", styles['Normal']))
        elements.append(Paragraph(f"Radius: {model_info['radius']}", styles['Normal']))
        elements.append(Paragraph(f"nBits: {model_info['nBits']}", styles['Normal']))
        elements.append(Paragraph(f"Algorithm: {model_info['algorithm']}", styles['Normal']))
        elements.append(Spacer(1, 12))

    # Convert headers to Paragraphs for wrapping
    header_paragraphs = [Paragraph(header, styles['Normal']) for header in headers]

    # Convert SMILES strings to Paragraphs for wrapping
    for i in range(len(table_data)):
        table_data[i][0] = Paragraph(table_data[i][0], styles['Normal'])

    # Add prediction table
    data = [header_paragraphs] + table_data
    table = Table(data, repeatRows=1)

    # Apply style to table
    table.setStyle(TableStyle([
#        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    # Ensure SMILES strings can wrap
    table._argW[0] = 2.5 * inch  # width of SMILES column
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

@app.route('/api', methods=['POST'])
def apipredict():
    data = request.json
    smiles = data.get('smiles', [])
    models = data.get('models', [])
    output_format = data.get('format', 'json')  # Default to JSON format

    if not smiles or not isinstance(smiles, list):
        return jsonify({'error': 'Invalid input: please provide a list of SMILES strings.'}), 400

    if not models or not isinstance(models, list):
        return jsonify({'error': 'Invalid input: please provide a list of model names.'}), 400

    all_predictions = {}

    for model_name in models:
        model_path = os.path.join(MODELS_DIR, model_name, f"{model_name}_meta.json")
        if not os.path.exists(model_path):
            return jsonify({'error': f"Model {model_name} does not exist."}), 400

        model = SklearnModel.fromFile(model_path)
        predictions = model.predictMols(smiles)
        predictions_formatted = [f"{pred[0]:.4f}" for pred in predictions]
        all_predictions[model_name] = predictions_formatted

    # Format the result
    result = []
    for i, smile in enumerate(smiles):
        result_entry = {'smiles': smile}
        for model_name in models:
            result_entry[f'prediction ({model_name})'] = all_predictions[model_name][i]
        result.append(result_entry)

    if output_format == 'text':
        result_text = '\n'.join([f"SMILES: {entry['smiles']} -> " + ", ".join([f"{model}: {pred}" for model, pred in entry.items() if model != 'smiles']) for entry in result])
        return Response(result_text, mimetype='text/plain')

    if output_format == 'csv':
        # Create an in-memory output file for CSV data
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
        output.seek(0)
        return Response(output.getvalue(), mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=predictions.csv"})


    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
