class SpeechTranslator {
    constructor() {
        this.isRecording = false;
        this.socket = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        this.startButton = document.getElementById('startButton');
        this.stopButton = document.getElementById('stopButton');
        this.sourceText = document.getElementById('sourceText');
        this.translatedText = document.getElementById('translatedText');
        this.status = document.getElementById('status');
        this.audioFile = document.getElementById('audioFile');
        this.sourceLanguage = document.getElementById('sourceLanguage');
        this.targetLanguage = document.getElementById('targetLanguage');
    }

    attachEventListeners() {
        this.startButton.addEventListener('click', () => this.startRecording());
        this.stopButton.addEventListener('click', () => this.stopRecording());
        this.audioFile.addEventListener('change', (e) => this.handleFileUpload(e));
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstart = () => {
                this.isRecording = true;
                this.updateUI();
                this.connectWebSocket();
            };

            this.mediaRecorder.onstop = () => {
                this.isRecording = false;
                this.updateUI();
                if (this.socket) {
                    this.socket.close();
                }
            };

            this.mediaRecorder.start(1000); // Collect data every second
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showError('Could not access microphone');
        }
    }

    stopRecording() {
        if (this.mediaRecorder) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/stream`;
        this.socket = new WebSocket(wsUrl);

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateTranslation(data);
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Connection error');
        };
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            this.updateTranslation(data);
        } catch (error) {
            console.error('Error uploading file:', error);
            this.showError('Error processing file');
        }
    }

    updateTranslation(data) {
        this.sourceText.textContent = data.source_text;
        this.translatedText.textContent = data.translated_text;
    }

    updateUI() {
        this.startButton.disabled = this.isRecording;
        this.stopButton.disabled = !this.isRecording;
        this.status.textContent = this.isRecording ? 'Recording...' : '';
    }

    showError(message) {
        this.status.textContent = `Error: ${message}`;
        this.status.style.color = 'var(--error-color)';
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new SpeechTranslator();
});
