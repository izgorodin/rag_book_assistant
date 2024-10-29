// Функции для управления прогресс-баром
function updateProgress(label, current, total) {
    const container = document.querySelector('.progress-container');
    const labelEl = container.querySelector('.progress-label');
    const fillEl = container.querySelector('.progress-fill');
    
    container.style.display = 'block';
    labelEl.textContent = `${label}: ${current}/${total}`;
    fillEl.style.width = `${(current/total) * 100}%`;
    
    if (current >= total) {
        setTimeout(() => {
            container.style.display = 'none';
        }, 1000);
    }
} 