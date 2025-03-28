const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const processedImage = document.getElementById('processedImage');
const gestureText = document.getElementById('gesture');
const ctx = canvas.getContext('2d');
let isProcessing = false;

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 1280,
                height: 720,
                frameRate: { ideal: 30 }
            } 
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera:", err);
    }
}

async function processFrame() {
    if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
        isProcessing = true;
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0);
        
        // Get canvas data as base64 image
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            processedImage.src = result.image;
            
            // Update the gesture and distance information
            document.getElementById('gesture').textContent = result.gesture;
            document.getElementById('distance').textContent = result.distance;
        } catch (err) {
            console.error("Error processing frame:", err);
        } finally {
            isProcessing = false;
        }
    }
    
    requestAnimationFrame(processFrame);
}

// Initialize
initCamera().then(() => {
    video.addEventListener('play', () => {
        processFrame();
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const volumeSlider = document.getElementById('volumeSlider');
    const volumeDisplay = document.getElementById('volume');
    const volumeUpBtn = document.getElementById('volumeUp');
    const volumeDownBtn = document.getElementById('volumeDown');

    async function updateVolume(value) {
        try {
            const response = await fetch('/set_volume', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ volume: value / 100 })
            });
            const result = await response.json();
            volumeDisplay.textContent = Math.round(result.volume * 100);
        } catch (err) {
            console.error("Error updating volume:", err);
        }
    }

    volumeSlider.addEventListener('input', (e) => {
        updateVolume(e.target.value);
    });

    volumeUpBtn.addEventListener('click', () => {
        const newValue = Math.min(100, parseInt(volumeSlider.value) + 10);
        volumeSlider.value = newValue;
        updateVolume(newValue);
    });

    volumeDownBtn.addEventListener('click', () => {
        const newValue = Math.max(0, parseInt(volumeSlider.value) - 10);
        volumeSlider.value = newValue;
        updateVolume(newValue);
    });
});