
import os
import cv2
import torch
from gfpgan.utils import GFPGANer
from flask import Flask, request, jsonify, send_file
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
import base64 

model_realesr = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path_realesr = 'realesr-general-x4v3.pth'

# Background enhancer with RealESRGAN
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

model_gfpgan_1_4 = GFPGANer(model_path='GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)


os.makedirs('output', exist_ok=True)


# def inference(img, version, scale, weight):
def inference(img, version, scale):
    # weight /= 100
    print(img, version, scale)
    try:
        extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:  # for gray inputs
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == 'v1.4':
            face_enhancer = GFPGANer(
            model_path='GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
      
        try:
            # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        else:
            extension = 'jpg'
        save_path = f'output/out.{extension}'
        cv2.imwrite(save_path, output)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print('global exception', error)
        return None, None

app = Flask(__name__)

@app.route('/reconstruir', methods=['POST'])
def reconstruir_imagem():
    try:

        token = request.form.get('token')
        version = request.form.get('version',"v1.4")
        scale = int(request.form.get('scale',2))
        img_file = request.files['imagem']

        if token == "api_key_abcd":

            temp_filename = 'temp.jpg'
            img_file.save(temp_filename)

            # output, save_path = inference(temp_filename, version, scale)
            output, save_path = inference(temp_filename, version, scale)

            if output is not None:
                # return send_file(save_path, mimetype='image/jpeg')
                 with open(save_path, 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    return jsonify({'status': 'success', 'message':'Imagem restaurada com sucesso', 'image_base64': encoded_image})
            else:
                return jsonify({'status': 'error', 'message':'Falha na reconstrução da imagem'})

           
        else:

            return jsonify({'status': 'error', 'message': 'Token invalido'})

       
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
