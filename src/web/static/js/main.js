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

    // Настройка marked для использования Prism
    marked.setOptions({
        highlight: function(code, lang) {
            if (Prism.languages[lang]) {
                return Prism.highlight(code, Prism.languages[lang], lang);
            }
            return code;
        },
        breaks: true,
        gfm: true
    });

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

    // Единая функция для показа уведомлений
    function showNotification(message, isError = false) {
        let notification = document.querySelector('.notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.className = 'notification';
            document.body.appendChild(notification);
        }
        
        notification.textContent = message;
        notification.className = `notification ${isError ? 'error' : ''}`;
        notification.classList.add('show');
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
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
        if (file.size > 100 * 1024 * 1024) { 
            uploadStatus.textContent = 'File size too large (max 100MB)';
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

    // Обновленная функция добавления кнопок копирования
    function addCodeCopyButtons() {
        const codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(codeBlock => {
            // Проверяем, нет ли уже кнопки
            if (codeBlock.parentElement.querySelector('.copy-button')) return;
            
            const container = codeBlock.parentElement;
            container.style.position = 'relative';
            
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.innerHTML = `
                <svg viewBox="0 0 24 24">
                    <path d="M16 1H4C2.9 1 2 1.9 2 3v14h2V3h12V1zm3 4H8C6.9 5 6 5.9 6 7v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                </svg>
            `;
            
            copyButton.addEventListener('click', async (e) => {
                e.stopPropagation(); // Предотвращаем всплытие события
                const code = codeBlock.textContent;
                
                try {
                    await navigator.clipboard.writeText(code);
                    showNotification('Code copied to clipboard!');
                    
                    // Меняем иконку на галочку
                    copyButton.innerHTML = `
                        <svg viewBox="0 0 24 24">
                            <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
                        </svg>
                    `;
                    
                    setTimeout(() => {
                        copyButton.innerHTML = `
                            <svg viewBox="0 0 24 24">
                                <path d="M16 1H4C2.9 1 2 1.9 2 3v14h2V3h12V1zm3 4H8C6.9 5 6 5.9 6 7v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                            </svg>
                        `;
                    }, 2000);
                } catch (err) {
                    showNotification('Failed to copy code', true);
                }
            });
            
            container.appendChild(copyButton);
        });
    }

    // Единая функция отображения ответа
    function displayAnswer(markdown) {
        const answerStatus = document.getElementById('answerStatus');
        answerStatus.innerHTML = marked(markdown);
        
        // Ждем следующего тика для обработки кода
        setTimeout(() => {
            Prism.highlightAll();
            addCodeCopyButtons();
        }, 0);
    }

    // Добавляем обработчик загрузки документа
    document.addEventListener('DOMContentLoaded', () => {
        // Инициализация при загрузке страницы
        Prism.highlightAll();
        addCodeCopyButtons();
    });
});
