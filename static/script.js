// カメラ映像を表示するビデオタグ
const video = document.getElementById('video');

// 1. Webカメラを起動する
navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => { alert("Camera Error: " + err); });

// 2. 撮影ボタンが押された時の動き
document.getElementById('snap').addEventListener('click', async () => {
    // 画面の表示切り替え（ロード中を表示）
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result_area').classList.add('hidden');

    // Canvasを使って映像を画像データに変換
    const canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');

    // Python(Flask)サーバーに画像を送信
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
        } else {
            // 結果を表示
            document.getElementById('result_img').src = data.image;
            document.getElementById('result_text').innerText = data.text;
            
            // 音声を再生
            const audio = document.getElementById('audio_player');
            audio.src = "data:audio/mp3;base64," + data.audio;
            audio.play().catch(e => console.log("Auto-play blocked"));
            
            document.getElementById('result_area').classList.remove('hidden');
        }
    } catch (e) {
        alert("Network Error");
    } finally {
        // ロード中を消す
        document.getElementById('loading').style.display = 'none';
    }
});
