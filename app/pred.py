import base64
import io
import json
import logging
import os

from rdkit import Chem
from rdkit.Chem import Draw


# define RDKit image implementer
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

# define invalid SMILES scrubber
def validate_smiles(smiles_list):
    """
    Validates a list of SMILES strings. Returns a list of invalid SMILES.
    """
    invalid_smiles = []
    for smile in smiles_list:
        if Chem.MolFromSmiles(smile) is None:
            invalid_smiles.append(smile)
    return invalid_smiles
    
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

