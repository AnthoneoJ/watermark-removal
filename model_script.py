VERSION = "1.2"
"""
- resize image so that aspect ratio match mask

Source: https://github.com/zuruoke/watermark-removal
https://github.com/AnthoneoJ/watermark-removal
"""
import os, shutil, copy
from PIL import Image

import gdown
import cv2
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
from preprocess_image import preprocess_image

class ModelHandler:
    def __init__(self, use_gpu: bool = True) -> None:
        if use_gpu:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only

        # Download models
        self.model_dir = os.path.abspath('model')
        if not self.is_model_downloaded():
            url = 'https://drive.google.com/drive/folders/1xRV4EdjJuAfsX9pQme6XeoFznKXG0ptJ?usp=sharing'
            temp_dir = 'model_temp'
            gdown.download_folder(url, output=temp_dir)
            for filename in os.listdir(temp_dir):
                source_file = os.path.join(temp_dir, filename)
                destination_file = os.path.join(self.model_dir, filename)
                shutil.copy(source_file, destination_file)
            shutil.rmtree(temp_dir)

        self.FLAGS = ng.Config('inpaint.yml')
        self.model = InpaintCAModel()

    def get_prediction(self, input_data: dict) -> str:
        image: Image.Image = input_data["input_image"]
        watermark_type: str = input_data["input_text"]
        if watermark_type=="placeholder" or watermark_type=="":
            watermark_type = "istock"
            
        mask_image_size = (683, 1024, 3) # based on utils/istock/landscape/mask.png
        mask_aspect_ratio = mask_image_size[0] / mask_image_size[1]
        # Get the current size of the input image
        input_width, input_height = image.size
        input_aspect_ratio = input_width / input_height
        # Resize the image to match the aspect ratio of the mask image
        if input_aspect_ratio > mask_aspect_ratio:
            # Input image is wider than the mask aspect ratio
            new_width = int(mask_image_size[1] * input_aspect_ratio)
            new_height = mask_image_size[1]
        else:
            # Input image is taller than or matches the mask aspect ratio
            new_width = mask_image_size[0]
            new_height = int(mask_image_size[0] / input_aspect_ratio)
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            
        input_image = preprocess_image(image, watermark_type)

        output_img = copy.copy(input_image)
        tf.compat.v1.reset_default_graph()
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        if (input_image.shape != (0,)):
            with tf.compat.v1.Session(config=sess_config) as sess:
                input_image = tf.constant(input_image, dtype=tf.float32)
                output = self.model.build_server_graph(self.FLAGS, input_image)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                # load pretrained model
                vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.train.load_variable(self.model_dir, from_name)
                    assign_ops.append(tf.compat.v1.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded.')
                result = sess.run(output)
                output_img = Image.fromarray(result[0][:, :, ::-1])
                
        output_img = output_img.resize((input_width, input_height), Image.Resampling.BICUBIC)

        return output_img

    def is_model_downloaded(self) -> bool:
        part0_exists = os.path.exists(os.path.join(self.model_dir, 'checkpoint'))
        part1_exists = os.path.exists(os.path.join(self.model_dir, 'snap-0.data-00000-of-00001'))
        part2_exists = os.path.exists(os.path.join(self.model_dir, 'snap-0.index'))
        part3_exists = os.path.exists(os.path.join(self.model_dir, 'snap-0.meta'))

        return part0_exists and part1_exists and part2_exists and part3_exists
    
# Example usage
if __name__ == '__main__':
    model = ModelHandler()

    from IPython.display import display
    import urllib.request
    url = "https://user-images.githubusercontent.com/51057490/140277713-c7d6e2b9-db62-4793-823a-25ed0c4e2771.png"
    urllib.request.urlretrieve(url,"_temp.png")
    img = Image.open("_temp.png")
    watermark_type = "istock"
    input_data = {
        "input_image": img,
        "input_text": watermark_type
    }
    output = model.get_prediction(input_data)
    print(type(output))
    display(output)