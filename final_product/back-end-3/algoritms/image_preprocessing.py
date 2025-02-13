from PIL import Image
import numpy as np
from io import BytesIO

def preprocess_image(image_bytes):
    """Preprocess the image for face recognition."""
    try:
        image_pil = Image.open(BytesIO(image_bytes))

        # Convert RGBA images to RGB
        if image_pil.mode == 'RGBA':
            image_pil = image_pil.convert('RGB')
        
        # Resize image to standard size for better performance
        image_pil = image_pil.resize((400, 400))

        image_np = np.array(image_pil)
        return image_np
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None
