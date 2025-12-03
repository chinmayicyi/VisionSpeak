// ========================================
// VisionSpeak - Frontend Controller
// ========================================

class VisionSpeakApp {
    constructor() {
        // State
        this.isDetecting = false;
        this.isMuted = false;
        this.currentMode = 'live';
        this.videoUpdateInterval = null;
        this.statsUpdateInterval = null;
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        
        // DOM Elements
        this.elements = {
            // Mode toggle
            liveModeBtn: document.getElementById('liveModeBtn'),
            uploadModeBtn: document.getElementById('uploadModeBtn'),
            liveMode: document.getElementById('liveMode'),
            uploadMode: document.getElementById('uploadMode'),
            
            // Video (Live Mode)
            videoFeed: document.getElementById('videoFeed'),
            videoPlaceholder: document.getElementById('videoPlaceholder'),
            videoContainer: document.getElementById('videoContainer'),
            liveBadge: document.getElementById('liveBadge'),
            objectCount: document.getElementById('objectCount'),
            
            // Stats
            personCount: document.getElementById('personCount'),
            otherCount: document.getElementById('otherCount'),
            fpsCount: document.getElementById('fpsCount'),
            
            // Controls
            startStopBtn: document.getElementById('startStopBtn'),
            muteBtn: document.getElementById('muteBtn'),
            snapshotBtn: document.getElementById('snapshotBtn'),
            forceSpeakBtn: document.getElementById('forceSpeakBtn'),
            
            // Upload
            uploadDropzone: document.getElementById('uploadDropzone'),
            fileInput: document.getElementById('fileInput'),
            uploadProgress: document.getElementById('uploadProgress'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            uploadResult: document.getElementById('uploadResult'),
            resultImage: document.getElementById('resultImage'),
            resultVideo: document.getElementById('resultVideo'),
            uploadAnotherBtn: document.getElementById('uploadAnotherBtn'),
            
            // Settings
            confidenceSlider: document.getElementById('confidenceSlider'),
            confidenceValue: document.getElementById('confidenceValue'),
            volumeSlider: document.getElementById('volumeSlider'),
            volumeValue: document.getElementById('volumeValue'),
            rateSlider: document.getElementById('rateSlider'),
            rateValue: document.getElementById('rateValue'),
            voiceSelect: document.getElementById('voiceSelect'),
            
            // Terminal
            terminalBody: document.getElementById('terminalBody'),
            clearLogBtn: document.getElementById('clearLogBtn'),
            
            // Theme
            themeToggle: document.getElementById('themeToggle'),
            
            // Toast
            toast: document.getElementById('toast')
        };
        
        this.init();
    }
    
    // ========================================
    // Initialization
    // ========================================
    
    init() {
        this.setupEventListeners();
        this.loadTheme();
        this.log('System ready. Select Live Camera or Upload Media.', 'welcome');
    }
    
    setupEventListeners() {
        // Mode toggle
        this.elements.liveModeBtn.addEventListener('click', () => this.switchMode('live'));
        this.elements.uploadModeBtn.addEventListener('click', () => this.switchMode('upload'));
        
        // Control buttons
        this.elements.startStopBtn.addEventListener('click', () => this.toggleDetection());
        this.elements.muteBtn.addEventListener('click', () => this.toggleMute());
        this.elements.snapshotBtn.addEventListener('click', () => this.takeSnapshot());
        this.elements.forceSpeakBtn.addEventListener('click', () => this.forceSpeak());
        
        // Upload
        this.elements.uploadDropzone.addEventListener('click', () => this.elements.fileInput.click());
        this.elements.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.elements.uploadAnotherBtn.addEventListener('click', () => this.resetUpload());
        
        // Drag and drop
        this.elements.uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadDropzone.classList.add('drag-over');
        });
        
        this.elements.uploadDropzone.addEventListener('dragleave', () => {
            this.elements.uploadDropzone.classList.remove('drag-over');
        });
        
        this.elements.uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadDropzone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
        
        // Settings
        this.elements.confidenceSlider.addEventListener('input', (e) => this.updateConfidence(e.target.value));
        this.elements.volumeSlider.addEventListener('input', (e) => this.updateVolume(e.target.value));
        this.elements.rateSlider.addEventListener('input', (e) => this.updateRate(e.target.value));
        this.elements.voiceSelect.addEventListener('change', (e) => this.updateVoice(e.target.value));
        
        // Terminal
        this.elements.clearLogBtn.addEventListener('click', () => this.clearLog());
        
        // Theme
        this.elements.themeToggle.addEventListener('click', () => this.toggleTheme());
    }
    
    // ========================================
    // Mode Switching
    // ========================================
    
    switchMode(mode) {
        // Stop detection if switching from live mode
        if (this.currentMode === 'live' && this.isDetecting) {
            this.stopDetection();
        }
        
        this.currentMode = mode;
        
        if (mode === 'live') {
            this.elements.liveModeBtn.classList.add('active');
            this.elements.uploadModeBtn.classList.remove('active');
            this.elements.liveMode.classList.add('active');
            this.elements.uploadMode.classList.remove('active');
            this.log('Switched to Live Camera mode', 'info');
        } else {
            this.elements.liveModeBtn.classList.remove('active');
            this.elements.uploadModeBtn.classList.add('active');
            this.elements.liveMode.classList.remove('active');
            this.elements.uploadMode.classList.add('active');
            this.log('Switched to Upload Media mode', 'info');
        }
    }
    
    // ========================================
    // Detection Control (Live Mode)
    // ========================================
    
    async toggleDetection() {
        if (this.isDetecting) {
            await this.stopDetection();
        } else {
            await this.startDetection();
        }
    }
    
    async startDetection() {
        try {
            this.log('Starting live detection...', 'info');
            
            const response = await fetch('/api/start', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'started' || data.status === 'success') {
                this.isDetecting = true;
                this.updateUIState(true);
                this.startVideoFeed();
                this.startStatsUpdate();
                this.log('Live detection started!', 'success');
                this.showToast('Detection started', 'success');
            } else {
                throw new Error(data.message || 'Failed to start detection');
            }
        } catch (error) {
            console.error('Start detection error:', error);
            this.log(`Error: ${error.message}`, 'error');
            this.showToast('Failed to start detection', 'error');
        }
    }
    
    async stopDetection() {
        try {
            this.log('Stopping detection...', 'info');
            
            const response = await fetch('/api/stop', { method: 'POST' });
            const data = await response.json();
            
            this.isDetecting = false;
            this.updateUIState(false);
            this.stopVideoFeed();
            this.stopStatsUpdate();
            this.log('Detection stopped.', 'info');
            this.showToast('Detection stopped', 'info');
        } catch (error) {
            console.error('Stop detection error:', error);
            this.log(`Error: ${error.message}`, 'error');
            this.showToast('Failed to stop detection', 'error');
        }
    }
    
    updateUIState(isActive) {
        if (isActive) {
            this.elements.startStopBtn.classList.add('active');
            this.elements.startStopBtn.innerHTML = '<i class="fas fa-stop"></i><span>Stop Detection</span>';
            
            this.elements.muteBtn.disabled = false;
            this.elements.snapshotBtn.disabled = false;
            this.elements.forceSpeakBtn.disabled = false;
            
            this.elements.confidenceSlider.disabled = false;
            this.elements.volumeSlider.disabled = false;
            this.elements.rateSlider.disabled = false;
            this.elements.voiceSelect.disabled = false;
            
            this.elements.videoContainer.classList.add('active');
            this.elements.liveBadge.classList.add('active');
            this.elements.liveBadge.innerHTML = '<span class="pulse-dot"></span><span>LIVE</span>';
        } else {
            this.elements.startStopBtn.classList.remove('active');
            this.elements.startStopBtn.innerHTML = '<i class="fas fa-play"></i><span>Start Detection</span>';
            
            this.elements.muteBtn.disabled = true;
            this.elements.snapshotBtn.disabled = true;
            this.elements.forceSpeakBtn.disabled = true;
            
            this.elements.confidenceSlider.disabled = true;
            this.elements.volumeSlider.disabled = true;
            this.elements.rateSlider.disabled = true;
            this.elements.voiceSelect.disabled = true;
            
            this.elements.videoContainer.classList.remove('active');
            this.elements.liveBadge.classList.remove('active');
            this.elements.liveBadge.innerHTML = '<span class="pulse-dot"></span><span>STANDBY</span>';
        }
    }
    
    // ========================================
    // Video Feed
    // ========================================
    
    startVideoFeed() {
        this.elements.videoFeed.src = '/video_feed?' + new Date().getTime();
        this.elements.videoFeed.classList.add('active');
        this.elements.videoPlaceholder.classList.add('hidden');
        
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        
        this.elements.videoFeed.onload = () => {
            this.frameCount++;
        };
    }
    
    stopVideoFeed() {
        this.elements.videoFeed.src = '';
        this.elements.videoFeed.classList.remove('active');
        this.elements.videoPlaceholder.classList.remove('hidden');
        
        this.elements.objectCount.textContent = '0 Objects';
        this.elements.personCount.textContent = '0';
        this.elements.otherCount.textContent = '0';
        this.elements.fpsCount.textContent = '0';
    }
    
    // ========================================
    // Stats Update
    // ========================================
    
    startStatsUpdate() {
        this.statsUpdateInterval = setInterval(() => this.fetchStats(), 1000);
    }
    
    stopStatsUpdate() {
        if (this.statsUpdateInterval) {
            clearInterval(this.statsUpdateInterval);
            this.statsUpdateInterval = null;
        }
    }
    
    async fetchStats() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.running) {
                const totalCount = data.object_count || 0;
                this.elements.objectCount.textContent = `${totalCount} Object${totalCount !== 1 ? 's' : ''}`;
                
                const objects = data.objects || [];
                const personCount = objects.filter(obj => obj.name === 'person').length;
                const otherCount = totalCount - personCount;
                
                this.elements.personCount.textContent = personCount;
                this.elements.otherCount.textContent = otherCount;
                
                const now = Date.now();
                const elapsed = (now - this.lastFrameTime) / 1000;
                if (elapsed > 0) {
                    const fps = Math.round(this.frameCount / elapsed);
                    this.elements.fpsCount.textContent = fps;
                    this.frameCount = 0;
                    this.lastFrameTime = now;
                }
                
                if (objects.length > 0 && Math.random() < 0.05) {
                    const objectNames = objects.map(obj => obj.name).join(', ');
                    this.log(`Detected: ${objectNames}`, 'info');
                }
            }
        } catch (error) {
            console.error('Stats fetch error:', error);
        }
    }
    
    // ========================================
    // Audio Control
    // ========================================
    
    async toggleMute() {
        try {
            const response = await fetch('/api/toggle_narration', { method: 'POST' });
            const data = await response.json();
            
            this.isMuted = data.status === 'paused';
            
            if (this.isMuted) {
                this.elements.muteBtn.classList.add('muted');
                this.elements.muteBtn.innerHTML = '<i class="fas fa-volume-xmark"></i><span>Unmute</span>';
                this.log('Narration muted', 'warning');
                this.showToast('Narration muted', 'info');
            } else {
                this.elements.muteBtn.classList.remove('muted');
                this.elements.muteBtn.innerHTML = '<i class="fas fa-volume-up"></i><span>Mute</span>';
                this.log('Narration unmuted', 'success');
                this.showToast('Narration active', 'success');
            }
        } catch (error) {
            console.error('Toggle mute error:', error);
            this.log(`Error toggling mute: ${error.message}`, 'error');
        }
    }
    
    async forceSpeak() {
        try {
            await fetch('/api/force_speak', { method: 'POST' });
            this.log('Force speak triggered', 'info');
            this.showToast('Speaking current scene', 'info');
        } catch (error) {
            console.error('Force speak error:', error);
            this.log(`Error: ${error.message}`, 'error');
        }
    }
    
    // ========================================
    // Snapshot
    // ========================================
    
    async takeSnapshot() {
        try {
            this.log('Taking snapshot...', 'info');
            
            const response = await fetch('/api/snapshot', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.log(`Snapshot saved: ${data.filename}`, 'success');
                this.showToast('Snapshot saved!', 'success');
            } else {
                throw new Error('Snapshot failed');
            }
        } catch (error) {
            console.error('Snapshot error:', error);
            this.log(`Snapshot error: ${error.message}`, 'error');
            this.showToast('Failed to save snapshot', 'error');
        }
    }
    
    // ========================================
    // Upload Functionality
    // ========================================
    
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFileUpload(file);
        }
    }
    
    async handleFileUpload(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'video/mp4', 'video/avi', 'video/quicktime'];
        if (!validTypes.includes(file.type)) {
            this.showToast('Invalid file type. Use JPG, PNG, MP4, AVI, or MOV.', 'error');
            this.log('Invalid file type uploaded', 'error');
            return;
        }
        
        // Show progress
        this.elements.uploadDropzone.style.display = 'none';
        this.elements.uploadProgress.style.display = 'block';
        this.elements.progressText.textContent = 'Uploading...';
        
        this.log(`Uploading ${file.name}...`, 'info');
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.elements.progressText.textContent = 'Processing complete!';
                this.log(`Processing complete: ${data.filename}`, 'success');
                
                // Show result
                setTimeout(() => {
                    this.showUploadResult(data);
                }, 500);
                
                this.showToast('Media processed successfully!', 'success');
            } else {
                throw new Error(data.error || 'Processing failed');
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            this.log(`Upload error: ${error.message}`, 'error');
            this.showToast('Upload failed', 'error');
            this.resetUpload();
        }
    }
    
    showUploadResult(data) {
        this.elements.uploadProgress.style.display = 'none';
        this.elements.uploadResult.style.display = 'block';
        
        if (data.file_type === 'image') {
            this.elements.resultImage.src = data.url;
            this.elements.resultImage.style.display = 'block';
            this.elements.resultVideo.style.display = 'none';
        } else {
            this.elements.resultVideo.src = data.url;
            this.elements.resultVideo.style.display = 'block';
            this.elements.resultImage.style.display = 'none';
        }
    }
    
    resetUpload() {
        this.elements.uploadDropzone.style.display = 'flex';
        this.elements.uploadProgress.style.display = 'none';
        this.elements.uploadResult.style.display = 'none';
        this.elements.fileInput.value = '';
        this.elements.progressFill.style.width = '0%';
        this.log('Ready for new upload', 'info');
    }
    
    // ========================================
    // Settings
    // ========================================
    
    async updateConfidence(value) {
        this.elements.confidenceValue.textContent = `${value}%`;
        
        try {
            await fetch('/api/change_confidence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: parseFloat(value) / 100 })
            });
            
            this.log(`Confidence threshold: ${value}%`, 'info');
        } catch (error) {
            console.error('Confidence update error:', error);
        }
    }
    
    updateVolume(value) {
        this.elements.volumeValue.textContent = `${value}%`;
        this.log(`Volume: ${value}%`, 'info');
    }
    
    updateRate(value) {
        this.elements.rateValue.textContent = `${value} wpm`;
        this.log(`Speech rate: ${value} wpm`, 'info');
    }
    
    updateVoice(value) {
        this.log(`Voice: ${value}`, 'info');
    }
    
    // ========================================
    // Terminal/Logging
    // ========================================
    
    log(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const line = document.createElement('div');
        line.className = `terminal-line ${type}`;
        line.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="message">${this.escapeHtml(message)}</span>
        `;
        
        this.elements.terminalBody.appendChild(line);
        this.elements.terminalBody.scrollTop = this.elements.terminalBody.scrollHeight;
        
        while (this.elements.terminalBody.children.length > 100) {
            this.elements.terminalBody.removeChild(this.elements.terminalBody.firstChild);
        }
    }
    
    clearLog() {
        this.elements.terminalBody.innerHTML = `
            <div class="terminal-line welcome">
                <span class="timestamp">[${new Date().toLocaleTimeString()}]</span>
                <span class="message">Log cleared.</span>
            </div>
        `;
        this.showToast('Log cleared', 'info');
    }
    
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
    
    // ========================================
    // Theme
    // ========================================
    
    toggleTheme() {
        const body = document.body;
        const isDark = !body.classList.contains('light-theme');
        
        if (isDark) {
            body.classList.add('light-theme');
            this.elements.themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            localStorage.setItem('theme', 'light');
            this.log('Light mode enabled', 'info');
        } else {
            body.classList.remove('light-theme');
            this.elements.themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
            localStorage.setItem('theme', 'dark');
            this.log('Dark mode enabled', 'info');
        }
    }
    
    loadTheme() {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            this.elements.themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        }
    }
    
    // ========================================
    // Toast Notifications
    // ========================================
    
    showToast(message, type = 'info') {
        this.elements.toast.textContent = message;
        this.elements.toast.className = `toast ${type} show`;
        
        setTimeout(() => {
            this.elements.toast.classList.remove('show');
        }, 3000);
    }
}

// ========================================
// Initialize App
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    window.visionSpeakApp = new VisionSpeakApp();
});