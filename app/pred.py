import base64
import io
import logging

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
    


