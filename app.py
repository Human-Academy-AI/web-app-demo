from flask import Flask, render_template, request, jsonify
import boto3
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

if 'AWS_SHARED_CREDENTIALS_FILE' not in os.environ:
  os.environ['AWS_SHARED_CREDENTIALS_FILE'] = '/content/.aws/credentials'
if 'AWS_CONFIG_FILE' not in os.environ:
  os.environ['AWS_CONFIG_FILE'] = '/content/.aws/config'

# --- ユーザー設定 ---
REGION_NAME = "ap-northeast-1"
# --------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        img_data_b64 = data['image'].split(',')[1]

        img_bytes = base64.b64decode(img_data_b64)
        image = Image.open(io.BytesIO(img_bytes))

        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w, _ = img_bgr.shape

        rekognition = boto3.client('rekognition',
            region_name=REGION_NAME
        )
        polly = boto3.client('polly',
            region_name=REGION_NAME
        )

        response = rekognition.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=20, MinConfidence=50
        )

        labels = response['Labels']

        # --- 診断レポート作成 ---
        debug_lines = []
        debug_lines.append(f"📸 画像サイズ: 横{img_w}px / 縦{img_h}px")
        debug_lines.append("----------------------------")
        debug_lines.append("AWS解析結果トップ5:")

        speech_name = ""

        for i, label in enumerate(labels[:5]):
            name = label['Name']
            conf = label['Confidence']
            instances = label.get('Instances', [])

            if len(instances) > 0:
                status = "✅ 枠データあり"
                if speech_name == "": speech_name = name

                for instance in instances:
                    box = instance['BoundingBox']
                    x1 = int(box['Left'] * img_w)
                    y1 = int(box['Top'] * img_h)
                    x2 = int((box['Left'] + box['Width']) * img_w)
                    y2 = int((box['Top'] + box['Height']) * img_h)

                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(img_bgr, f"{name}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        status += "(座標異常)"
            else:
                status = "⚠️ 枠なし(概念のみ)"

            debug_lines.append(f"{i+1}. {name} ({conf:.1f}%) \n   -> {status}")

        result_text = "\n".join(debug_lines)

        # 音声
        if speech_name:
            speech_text = f"{speech_name}を見つけました"
        elif labels:
            speech_text = f"{labels[0]['Name']}だと思います"
        else:
            speech_text = "何もわかりませんでした"

        polly_res = polly.synthesize_speech(
            Text=speech_text, OutputFormat='mp3', VoiceId='Kazuha', Engine='neural'
        )
        audio_stream = polly_res['AudioStream'].read()
        audio_b64 = base64.b64encode(audio_stream).decode()

        _, buffer = cv2.imencode('.jpg', img_bgr)
        processed_img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        return jsonify({
            'image': processed_img_b64,
            'text': result_text,
            'audio': audio_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
