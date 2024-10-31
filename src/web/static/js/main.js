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

    // При отображении ответа
    function displayAnswer(markdown) {
        const answerStatus = document.getElementById('answerStatus');
        answerStatus.innerHTML = marked(markdown);
        
        // Подсвечиваем синтаксис
        Prism.highlightAll();
        
        // Добавляем кнопки копирования
        addCodeCopyButtons();
    }

    // Функция добавления кнопок копирования
    function addCodeCopyButtons() {
        console.log('Adding copy buttons...'); // Для отладки
        const codeBlocks = document.querySelectorAll('pre[class*="language-"]');
        console.log('Found code blocks:', codeBlocks.length); // Для отладки
        
        codeBlocks.forEach(block => {
            // Проверяем, нет ли уже кнопки
            if (block.querySelector('.code-copy-button')) {
                return;
            }
            
            // Создаем кнопку
            const copyButton = document.createElement('button');
            copyButton.className = 'code-copy-button';
            copyButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
            `;
            
            copyButton.addEventListener('click', async () => {
                const code = block.querySelector('code').textContent;
                try {
                    await navigator.clipboard.writeText(code);
                    copyButton.classList.add('copied');
                    
                    // Показываем уведомление
                    showNotification('Код скопирован!');
                    
                    setTimeout(() => {
                        copyButton.classList.remove('copied');
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                    showNotification('Ошибка копирования', true);
                }
            });
            
            block.style.position = 'relative';
            block.appendChild(copyButton);
        });
    }

    // Функция показа уведомления
    function showNotification(message, isError = false) {
        const notification = document.createElement('div');
        notification.className = `notification ${isError ? 'error' : ''}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        // Показываем уведомление
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Удаляем через 2 секунды
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 2000);
    }

    // Обновленная функция отображения ответа
    function displayAnswer(markdown) {
        const answerStatus = document.getElementById('answerStatus');
        answerStatus.innerHTML = marked(markdown);
        
        // Важно: ждем следующего тика, чтобы Prism успел обработать код
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
