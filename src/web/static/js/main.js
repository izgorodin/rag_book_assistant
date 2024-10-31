document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const progressContainer = document.querySelector('.progress-container');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    const progressLabel = document.querySelector('.progress-label');

    const questionForm = document.getElementById('questionForm');
    const answerStatus = document.getElementById('answerStatus');

    let ws = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    let currentAnswer = '';
    const copyButton = document.getElementById('copyAnswer');

    function connectWebSocket() {
        if (reconnectAttempts >= maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        console.log(`Attempting to connect to WebSocket at: ${wsUrl}`);

        try {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected successfully');
                reconnectAttempts = 0;
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'progress') {
                        updateProgress(data);
                        
                        if (data.answer) {
                            currentAnswer = data.answer;
                            answerStatus.innerHTML = marked.parse(data.answer);
                            copyButton.style.display = 'block';
                            progressContainer.style.display = 'none';
                        }
                        
                        if (data.error) {
                            answerStatus.textContent = `Error: ${data.error}`;
                            copyButton.style.display = 'none';
                            progressContainer.style.display = 'none';
                        }
                    }
                } catch (error) {
                    console.error('Error processing WebSocket message:', error);
                }
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                ws = null;
                reconnectAttempts++;
                setTimeout(connectWebSocket, 2000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            reconnectAttempts++;
            setTimeout(connectWebSocket, 2000);
        }
    }

    function updateProgress(data) {
        if (!progressContainer) return;
        
        progressContainer.style.display = 'block';
        const percent = (data.current / data.total) * 100;
        
        progressFill.style.width = `${percent}%`;
        progressText.textContent = `${Math.round(percent)}%`;
        progressLabel.textContent = data.status;
        
        if (percent >= 100) {
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 2000);
        }
    }

    function showNotification(message) {
        let notification = document.querySelector('.notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.className = 'notification';
            document.body.appendChild(notification);
        }
        
        notification.textContent = message;
        notification.classList.add('show');
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 2000);
    }

    // Обработчик копирования ответа
    copyButton.addEventListener('click', () => {
        if (!currentAnswer) return;
        
        const textarea = document.createElement('textarea');
        textarea.value = currentAnswer;
        document.body.appendChild(textarea);
        textarea.select();
        
        try {
            document.execCommand('copy');
            showNotification('Copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy:', err);
        }
        
        document.body.removeChild(textarea);
    });

    // Инициализация WebSocket
    connectWebSocket();

    // Обработка формы загрузки
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.querySelector('input[type="file"]');
        if (!fileInput.files.length) {
            uploadStatus.textContent = 'Please select a file';
            return;
        }
        
        const file = fileInput.files[0];
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            uploadStatus.textContent = 'File size too large (max 10MB)';
            return;
        }
        
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '0%';
        progressLabel.textContent = 'Starting upload...';
        
        const formData = new FormData(uploadForm);
        uploadStatus.textContent = 'Uploading...';
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            uploadStatus.textContent = result.message;
            
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            resetProgress();
            console.error('Upload error:', error);
        }
    });

    // Обработка формы вопроса
    questionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(questionForm);
        const question = formData.get('question');
        if (!question) return;
        
        answerStatus.textContent = 'Thinking...';
        copyButton.style.display = 'none';
        
        try {
            const response = await fetch(`/ask?question=${encodeURIComponent(question)}`, {
                method: 'GET'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                currentAnswer = result.answer;
                answerStatus.innerHTML = marked.parse(result.answer);
                copyButton.style.display = 'block';
            } else {
                answerStatus.textContent = `Error: ${result.detail || 'Failed to get answer'}`;
                copyButton.style.display = 'none';
            }
        } catch (error) {
            answerStatus.textContent = `Error: ${error.message}`;
            copyButton.style.display = 'none';
        }
    });
});
