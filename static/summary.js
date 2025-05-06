// Add playback cursor functionality
document.querySelectorAll('.audio-container').forEach(container => {
    const audio = container.querySelector('audio');
    const cursor = container.querySelector('.playback-cursor');
    const spectrogram = container.querySelector('.spectrogram-container');
    
    if (!audio || !cursor || !spectrogram) return;
    
    // Initialize cursor position
    cursor.style.left = '0px';
    
    // Update cursor position during playback
    function updateCursor() {
        if (!audio.duration) return;  // Skip if duration is not available
        
        const progress = audio.currentTime / audio.duration;
        const position = progress * spectrogram.offsetWidth;
        cursor.style.left = `${position}px`;
        
        // Request next frame if playing
        if (!audio.paused) {
            requestAnimationFrame(updateCursor);
        }
    }
    
    // Handle playback events
    audio.addEventListener('play', () => {
        requestAnimationFrame(updateCursor);
    });
    
    audio.addEventListener('timeupdate', () => {
        if (audio.paused) {
            updateCursor();  // Update cursor even when paused
        }
    });
    
    // Handle seeking by clicking on spectrogram
    spectrogram.addEventListener('click', (e) => {
        const rect = spectrogram.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const progress = x / rect.width;
        
        if (progress >= 0 && progress <= 1) {  // Ensure valid range
            audio.currentTime = progress * audio.duration;
            cursor.style.left = `${x}px`;
        }
    });
    
    // Show current time on hover
    spectrogram.addEventListener('mousemove', (e) => {
        const rect = spectrogram.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const progress = x / rect.width;
        const timeInSeconds = progress * audio.duration;
        
        if (timeInSeconds >= 0 && timeInSeconds <= audio.duration) {
            const minutes = Math.floor(timeInSeconds / 60);
            const seconds = Math.floor(timeInSeconds % 60);
            spectrogram.title = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    });
});

// Method visibility management
const activeMethods = new Set();

function toggleMethod(methodName) {
    const allMethods = document.querySelectorAll('[data-method]');
    const selectedBtn = document.querySelector(`#btn-${methodName}`);
    
    if (selectedBtn.classList.contains('active')) {
        // Remove method from active set
        activeMethods.delete(methodName);
        selectedBtn.classList.remove('active', 'bg-blue-100');
    } else {
        // Add method to active set
        activeMethods.add(methodName);
        selectedBtn.classList.add('active', 'bg-blue-100');
    }

    // If no methods are active, show all
    if (activeMethods.size === 0) {
        allMethods.forEach(el => {
            el.style.display = '';
        });
        return;
    }

    // Hide/show elements based on active methods
    allMethods.forEach(el => {
        const method = el.getAttribute('data-method');
        if (method === 'original' || activeMethods.has(method)) {
            el.style.display = ''; // Show selected methods and original
        } else {
            el.style.display = 'none'; // Hide others
        }
    });
}

// Initialize tooltips
document.querySelectorAll('[data-tooltip]').forEach(element => {
    element.addEventListener('mouseenter', e => {
        const tooltip = document.createElement('div');
        tooltip.className = 'absolute z-50 p-2 bg-gray-900 text-white text-sm rounded shadow-lg';
        tooltip.textContent = e.target.dataset.tooltip;
        tooltip.style.top = `${e.target.offsetTop + e.target.offsetHeight + 5}px`;
        tooltip.style.left = `${e.target.offsetLeft}px`;
        document.body.appendChild(tooltip);
        
        e.target.addEventListener('mouseleave', () => tooltip.remove());
    });
});

// Add click-to-zoom functionality for spectrograms
document.querySelectorAll('.spectrogram-img').forEach(img => {
    img.addEventListener('click', (e) => {
        // Don't zoom if we're clicking for seeking
        if (e.target.parentElement.classList.contains('spectrogram-container')) {
            return;
        }
        
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50';
        const modalImg = document.createElement('img');
        modalImg.src = img.src;
        modalImg.className = 'max-w-[90vw] max-h-[90vh] object-contain';
        modal.appendChild(modalImg);
        
        modal.addEventListener('click', () => modal.remove());
        document.body.appendChild(modal);
    });
    img.style.cursor = 'zoom-in';
}); 