document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const questionForm = document.getElementById('questionForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const answerStatus = document.getElementById('answerStatus');
    const progressContainer = document.querySelector('.progress-container');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    const progressLabel = document.querySelector('.progress-label');
    const progressStatus = document.getElementById('progress-status');

    // Функция для сброса прогресс-бара
    function resetProgress() {
        progressContainer.style.display = 'none';
        progressFill.style.width = '0%';
        progressText.textContent = '0%';
        progressLabel.textContent = '';
        progressStatus.style.display = 'none';
        progressStatus.textContent = '';
    }

    // WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateProgress(data);
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        progressStatus.style.display = 'block';
        progressStatus.textContent = 'WebSocket connection error';
        setTimeout(resetProgress, 3000);
    };

    // Обновленная функция updateProgress
    function updateProgress(data) {
        if (data.status && data.current !== undefined && data.total !== undefined) {
            const percent = (data.current / data.total) * 100;
            progressContainer.style.display = 'block';
            progressFill.style.width = `${percent}%`;
            progressText.textContent = `${Math.round(percent)}%`;
            progressLabel.textContent = data.status;
            
            if (data.chunks_count !== undefined) {
                progressStatus.style.display = 'block';
                progressStatus.textContent = `Processed ${data.chunks_count} chunks`;
                setTimeout(resetProgress, 2000);
            }
            
            if (data.error) {
                progressStatus.style.display = 'block';
                progressStatus.textContent = `Error: ${data.error}`;
                setTimeout(resetProgress, 3000);
            }
        }
    }

    // File upload handling
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(uploadForm);
        uploadStatus.textContent = 'Uploading...';
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '0%';
        progressLabel.textContent = 'Starting upload...';
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                uploadStatus.textContent = result.message;
                // Если нет WebSocket обновлений, скрыть прогресс-бар
                if (!result.processing) {
                    setTimeout(resetProgress, 2000);
                }
            } else {
                uploadStatus.textContent = `Error: ${result.detail || 'Upload failed'}`;
                setTimeout(resetProgress, 3000);
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            setTimeout(resetProgress, 3000);
        }
    });

    // Question handling
    questionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const question = document.getElementById('question').value;
        const answerStatus = document.getElementById('answerStatus');
        
        answerStatus.className = 'loading';
        answerStatus.textContent = 'Processing your question...';
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `question=${encodeURIComponent(question)}`
            });
            
            const result = await response.json();
            
            if (response.ok) {
                answerStatus.className = 'answer-container markdown-body';
                answerStatus.innerHTML = marked.parse(result.answer);
            } else {
                answerStatus.className = 'error';
                answerStatus.textContent = `Error: ${result.detail || 'Failed to get answer'}`;
            }
        } catch (error) {
            answerStatus.className = 'error';
            answerStatus.textContent = `Error: ${error.message}`;
        }
    });
});
