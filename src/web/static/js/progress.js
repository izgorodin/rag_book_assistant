const socket = io();
const progressBar = document.getElementById('progress-bar');
const progressStatus = document.getElementById('progress-status');

// Показываем прогресс бар при начале загрузки
function showProgress() {
    progressBar.parentElement.style.display = 'block';
    progressStatus.style.display = 'block';
}

// Скрываем прогресс бар при завершении
function hideProgress() {
    setTimeout(() => {
        progressBar.parentElement.style.display = 'none';
        progressStatus.style.display = 'none';
    }, 2000);
}

socket.on('progress_update', (data) => {
    showProgress();
    
    progressBar.style.width = `${data.progress}%`;
    progressBar.setAttribute('aria-valuenow', data.progress);
    
    let statusText = `${data.status}: ${data.progress.toFixed(1)}%`;
    if (data.current && data.total) {
        statusText += ` (${data.current}/${data.total})`;
    }
    progressStatus.textContent = statusText;
    
    if (data.error) {
        progressStatus.classList.add('text-danger');
        progressStatus.textContent = `Error: ${data.error}`;
    }
    
    if (data.progress >= 100) {
        hideProgress();
    }
});
