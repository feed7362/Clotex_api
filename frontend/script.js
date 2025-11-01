// Глобальні змінні
let uploadedFile = null;
let processedLayers = [];
let layerImages = [];

// DOM елементи
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const colorCount = document.getElementById('colorCount');
const colorValue = document.getElementById('colorValue');
const tolerance = document.getElementById('tolerance');
const toleranceValue = document.getElementById('toleranceValue');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const previewCanvas = document.getElementById('previewCanvas');
const layerControls = document.getElementById('layerControls');
const layersContainer = document.getElementById('layersContainer');
const downloadAllBtn = document.getElementById('downloadAllBtn');

// Ініціалізація
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    console.log('🎨 Screen Print Separator готовий до роботи!');
    
    // Перевірка доступності API
    checkAPIHealth();
});

function initializeEventListeners() {
    // Завантаження файлу - клік по зоні завантаження
    uploadArea.addEventListener('click', (e) => {
        // Якщо клік не по самому input, то тригеримо клік на input
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag & Drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Слайдери
    colorCount.addEventListener('input', (e) => {
        colorValue.textContent = e.target.value;
    });
    
    tolerance.addEventListener('input', (e) => {
        toleranceValue.textContent = e.target.value;
    });
    
    // Кнопка обробки
    processBtn.addEventListener('click', processImage);
    
    // Завантажити всі шари
    downloadAllBtn.addEventListener('click', downloadAllLayers);
}

// Обробка файлів
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        uploadedFile = file;
        displayUploadedImage(file);
        processBtn.disabled = false;
        showNotification(`Файл "${file.name}" завантажено успішно!`, 'success');
    } else {
        showNotification('Будь ласка, виберіть файл зображення (PNG, JPG, JPEG)', 'error');
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        uploadedFile = file;
        fileInput.files = e.dataTransfer.files;
        displayUploadedImage(file);
        processBtn.disabled = false;
        showNotification(`Файл "${file.name}" завантажено успішно!`, 'success');
    } else {
        showNotification('Будь ласка, перетягніть файл зображення (PNG, JPG, JPEG)', 'error');
    }
}

function displayUploadedImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const uploadContent = document.getElementById('uploadContent');
        const previewDiv = document.getElementById('previewImage');
        
        // Сховати оригінальний контент
        uploadContent.classList.add('hidden');
        
        // Показати попередній перегляд
        previewDiv.classList.remove('hidden');
        previewDiv.innerHTML = `<img src="${e.target.result}" alt="Uploaded image">`;
        
        // Змінити стиль зони завантаження
        uploadArea.style.padding = '1rem';
    };
    reader.readAsDataURL(file);
}

// Обробка зображення
async function processImage() {
    if (!uploadedFile) {
        showNotification('Спочатку завантажте зображення!', 'error');
        return;
    }
    
    // Показати прогрес
    resultsSection.classList.add('hidden');
    progressSection.classList.remove('hidden');
    processBtn.disabled = true;
    
    try {
        // Створити FormData
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('n_colors', colorCount.value);
        formData.append('tolerance', tolerance.value);
        
        // Відправити запит до Flask API
        updateProgress(10, 'Завантаження зображення на сервер...');
        
        const response = await fetch('http://localhost:5000/process', {
            method: 'POST',
            mode: 'cors',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Помилка обробки зображення');
        }
        
        updateProgress(50, 'Обробка зображення моделлю...');
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error('Сервер повернув помилку');
        }
        
        updateProgress(80, 'Завантаження результатів...');
        
        // Зберегти дані
        processedLayers = data.layers;
        
        // Завантажити зображення шарів
        await loadLayerImages(data.layers);
        
        updateProgress(100, 'Готово!');
        
        showNotification(`Успішно створено ${data.layers.length} шарів!`, 'success');
        
        // Показати результати
        setTimeout(() => {
            progressSection.classList.add('hidden');
            displayResults();
        }, 500);
        
    } catch (error) {
        console.error('Помилка:', error);
        updateProgress(0, '❌ Помилка обробки: ' + error.message);
        showNotification('Помилка: ' + error.message, 'error');
        processBtn.disabled = false;
        
        setTimeout(() => {
            progressSection.classList.add('hidden');
        }, 3000);
    }
}

function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressText.textContent = text;
}

async function loadLayerImages(layers) {
    layerImages = [];
    
    for (const layer of layers) {
        const img = new Image();
        img.src = 'data:image/png;base64,' + layer.image_base64;
        
        await new Promise((resolve) => {
            img.onload = resolve;
        });
        
        layerImages.push(img);
    }
}

// Відображення результатів
function displayResults() {
    resultsSection.classList.remove('hidden');
    
    // Відобразити canvas з шарами
    drawPreviewCanvas();
    
    // Створити контроли шарів
    createLayerControls();
    
    // Створити картки окремих шарів
    createLayerCards();
}

function drawPreviewCanvas() {
    const ctx = previewCanvas.getContext('2d');
    
    if (layerImages.length === 0) return;
    
    // Встановити розмір canvas
    const firstImg = layerImages[0];
    previewCanvas.width = firstImg.width;
    previewCanvas.height = firstImg.height;
    
    // Очистити canvas
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    // Підрахувати кількість видимих шарів для правильної прозорості
    const visibleLayers = layerImages.filter((_, index) => {
        const checkbox = document.getElementById(`layer-${index}`);
        return !checkbox || checkbox.checked;
    });
    
    // Встановити прозорість залежно від кількості видимих шарів
    const opacity = visibleLayers.length > 1 ? 0.85 / Math.sqrt(visibleLayers.length) : 1;
    
    // Намалювати всі видимі шари з прозорістю
    layerImages.forEach((img, index) => {
        const checkbox = document.getElementById(`layer-${index}`);
        if (!checkbox || checkbox.checked) {
            ctx.globalAlpha = opacity;
            ctx.globalCompositeOperation = 'source-over';
            ctx.drawImage(img, 0, 0);
        }
    });
    
    // Повернути прозорість до нормальної
    ctx.globalAlpha = 1;
}

function createLayerControls() {
    layerControls.innerHTML = '';
    
    processedLayers.forEach((layer, index) => {
        const control = document.createElement('div');
        control.className = 'layer-control';
        
        control.innerHTML = `
            <input type="checkbox" id="layer-${index}" checked>
            <div class="layer-color-preview" style="background-color: ${layer.color_hex}"></div>
            <div class="layer-info">
                <div class="layer-name">Шар ${layer.layer_number}</div>
                <div class="layer-hex">${layer.color_hex}</div>
            </div>
        `;
        
        // Додати слухач для перемальовування canvas
        const checkbox = control.querySelector('input[type="checkbox"]');
        checkbox.addEventListener('change', drawPreviewCanvas);
        
        layerControls.appendChild(control);
    });
}

function createLayerCards() {
    layersContainer.innerHTML = '';
    
    processedLayers.forEach((layer, index) => {
        const card = document.createElement('div');
        card.className = 'layer-card';
        
        card.innerHTML = `
            <div class="layer-card-header">
                <div class="layer-card-title">
                    <div class="layer-color-preview" style="background-color: ${layer.color_hex}; width: 24px; height: 24px;"></div>
                    <span>Шар ${layer.layer_number}</span>
                </div>
                <button class="layer-download-btn" data-index="${index}">
                    ⬇ Завантажити
                </button>
            </div>
            <img src="data:image/png;base64,${layer.image_base64}" class="layer-image" alt="Layer ${layer.layer_number}">
        `;
        
        // Додати слухач для завантаження
        const downloadBtn = card.querySelector('.layer-download-btn');
        downloadBtn.addEventListener('click', () => downloadLayer(index));
        
        layersContainer.appendChild(card);
    });
}

// Завантаження шарів
function downloadLayer(index) {
    const layer = processedLayers[index];
    const link = document.createElement('a');
    link.href = 'data:image/png;base64,' + layer.image_base64;
    link.download = `layer_${layer.layer_number}_${layer.color_hex}.png`;
    link.click();
}

function downloadAllLayers() {
    processedLayers.forEach((layer, index) => {
        setTimeout(() => downloadLayer(index), index * 200);
    });
}

// Перевірка доступності API
async function checkAPIHealth() {
    try {
        const response = await fetch('http://localhost:5000/health', {
            method: 'GET',
            mode: 'cors'
        });
        if (response.ok) {
            console.log('✅ API сервер доступний');
            showNotification('API сервер підключено успішно!', 'success');
        } else {
            console.warn('⚠️ API сервер відповів з помилкою');
            showNotification('API сервер працює, але є попередження', 'error');
        }
    } catch (error) {
        console.error('❌ API сервер недоступний:', error);
        showNotification('⚠️ Не вдалося підключитися до API. Переконайтесь що Flask запущений: python app.py', 'error');
    }
}

// Показати сповіщення
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#6366f1'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Додати CSS анімації для сповіщень
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);