

class GabrielControlPanel {
    constructor() {
        
        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        
        const isDevelopment = window.location.port === '5069' || window.location.port === '5500' || 
                            window.location.port === '3000' || 
                            window.location.hostname === '127.0.0.1' || 
                            (window.location.hostname === 'localhost' && window.location.port !== '8000');
        
        let apiHost, wsHost;
        
        if (isDevelopment) {
            
            
            const hostname = window.location.hostname;
            apiHost = `${hostname}:8000`;
            wsHost = `${hostname}:8000`;
            console.log(`Development/WebUI mode - connecting to API at ${hostname}:8000`);
        } else {
            
            apiHost = window.location.host;
            wsHost = window.location.host;
        }
        
        this.apiUrl = `${protocol}//${apiHost}`;
        this.wsUrl = `${wsProtocol}//${wsHost}/api/chat/ws`;
        this.autoScroll = true;
        
        
        console.log('Gabriel Control Panel URLs:');
        console.log('API URL:', this.apiUrl);
        console.log('WebSocket URL:', this.wsUrl);
        console.log('Development mode:', isDevelopment);
        this.consoleOutput = document.getElementById('consoleOutput');
        this.yapModeEnabled = false;
        this.v2ModeEnabled = false;
        this.v2ModeAvailable = false;
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        
        this.userSpeechBuffer = [];
        this.userSpeechBufferTimeout = null;
        this.userSpeechGroupingDelay = 2000;
        
        
        this.personalities = [];
        this.currentPersonality = null;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkStatus();
        this.loadYapModeStatus();
        this.loadV2ModeStatus();
        this.loadPersonalities();
        this.loadVRChatControlsStatus();
        this.connectWebSocket();
        
        
        setInterval(() => this.checkStatus(), 30000);
        
        this.addConsoleMessage('system', 'Control panel initialized. Checking AI status...');
    }

    bindEvents() {
        
        document.getElementById('sendButton').addEventListener('click', () => this.sendMessage());
        
        
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.sendMessage();
            }
        });

        
        document.getElementById('messageInput').addEventListener('input', (e) => {
            const counter = document.getElementById('charCounter');
            counter.textContent = `${e.target.value.length}/1000`;
        });

        
        document.querySelectorAll('.quick-buttons .btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const message = e.target.closest('button').dataset.message;
                if (message) {
                    this.sendQuickMessage(message);
                }
            });
        });

        
        document.getElementById('clearConsole').addEventListener('click', () => this.clearConsole());
        document.getElementById('toggleAutoScroll').addEventListener('click', () => this.toggleAutoScroll());
        document.getElementById('showLegend').addEventListener('click', () => this.toggleLegend());

        
        document.getElementById('yapModeToggle').addEventListener('change', (e) => {
            this.toggleYapMode(e.target.checked);
        });

        
        document.getElementById('v2ModeToggle').addEventListener('change', (e) => {
            this.toggleV2Mode(e.target.checked);
        });
        
        
        document.getElementById('refreshPersonalities').addEventListener('click', () => {
            this.loadPersonalities();
        });
        
        
        document.getElementById('safeModeButton').addEventListener('click', () => {
            this.enableSafeMode();
        });
        
        document.getElementById('voiceToggle').addEventListener('change', (e) => {
            this.toggleVRChatVoice(e.target.checked);
        });
        
        
        document.getElementById('reconnectButton').addEventListener('click', () => {
            this.reconnectSession();
        });
        
        
        document.getElementById('freshStartButton').addEventListener('click', () => {
            this.freshStart();
        });
    }

    async checkStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            this.updateStatus(data.status === 'healthy' && data.session_active, data);
            
            
            const chatResponse = await fetch(`${this.apiUrl}/api/chat/status`);
            const chatData = await chatResponse.json();
            
            this.updateApiStatus(chatData.success);
            
            
            this.loadVRChatControlsStatus();
            
        } catch (error) {
            this.updateStatus(false);
            this.updateApiStatus(false);
            console.error('Status check failed:', error);
        }
    }

    updateStatus(isOnline, data = {}) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        if (isOnline) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'AI Online';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'AI Offline';
        }
    }

    updateApiStatus(isOnline) {
        const apiStatus = document.getElementById('apiStatus');
        
        if (isOnline) {
            apiStatus.className = 'api-status-badge online';
            apiStatus.textContent = 'Connected';
        } else {
            apiStatus.className = 'api-status-badge offline';
            apiStatus.textContent = 'Disconnected';
        }
    }

    async loadYapModeStatus() {
        try {
            
            
            
            this.yapModeEnabled = false;
            document.getElementById('yapModeToggle').checked = this.yapModeEnabled;
        } catch (error) {
            console.error('Failed to load yap mode status:', error);
        }
    }

    async loadV2ModeStatus() {
        try {
            
            const response = await fetch(`${this.apiUrl}/api/v2/status`);
            const data = await response.json();
            
            if (data.success) {
                this.v2ModeEnabled = data.v2_mode_enabled;
                this.v2ModeAvailable = data.v2_available;
                
                const toggle = document.getElementById('v2ModeToggle');
                toggle.checked = this.v2ModeEnabled;
                
                
                if (!this.v2ModeAvailable) {
                    toggle.disabled = true;
                    toggle.parentElement.parentElement.style.opacity = '0.5';
                    
                    
                    const description = toggle.parentElement.parentElement.querySelector('.control-description');
                    if (description) {
                        description.textContent = 'V2 mode is not available on this system';
                    }
                }
            } else {
                
                this.v2ModeEnabled = false;
                this.v2ModeAvailable = false;
                document.getElementById('v2ModeToggle').checked = this.v2ModeEnabled;
                document.getElementById('v2ModeToggle').disabled = true;
            }
        } catch (error) {
            console.error('Failed to load V2 mode status:', error);
            
            this.v2ModeEnabled = false;
            this.v2ModeAvailable = false;
            document.getElementById('v2ModeToggle').checked = this.v2ModeEnabled;
            document.getElementById('v2ModeToggle').disabled = true;
        }
    }

    async toggleYapMode(enabled) {
        try {
            this.showLoading();
            
            
            const message = enabled ? 
                'Enable yap mode - prevent interruptions during conversations' : 
                'Disable yap mode - allow interruptions during conversations';
            
            const response = await fetch(`${this.apiUrl}/api/chat/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    system_instruction: true,
                    turn_complete: true
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.yapModeEnabled = enabled;
                this.showToast('success', `Yap mode ${enabled ? 'enabled' : 'disabled'}`);
                this.addConsoleMessage('success', `Yap mode ${enabled ? 'enabled' : 'disabled'}`);
            } else {
                throw new Error(data.message || 'Failed to toggle yap mode');
            }
            
        } catch (error) {
            console.error('Failed to toggle yap mode:', error);
            this.showToast('error', `Failed to toggle yap mode: ${error.message}`);
            
            
            document.getElementById('yapModeToggle').checked = this.yapModeEnabled;
        } finally {
            this.hideLoading();
        }
    }

    async toggleV2Mode(enabled) {
        
        if (enabled && !this.v2ModeAvailable) {
            this.showToast('error', 'V2 mode is not available on this system');
            
            document.getElementById('v2ModeToggle').checked = this.v2ModeEnabled;
            return;
        }
        
        try {
            this.showLoading();
            
            
            const response = await fetch(`${this.apiUrl}/api/v2/toggle`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    enable_v2: enabled
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.v2ModeEnabled = enabled;
                const modeText = enabled ? 'V2' : 'V1';
                this.showToast('success', `Switched to ${modeText} mode`);
                this.addConsoleMessage('success', `AI mode switched to ${modeText}`);
                
                
                if (enabled) {
                    this.addConsoleMessage('info', 'V2 mode: Enhanced voice quality and advanced features');
                } else {
                    this.addConsoleMessage('info', 'V1 mode: Standard operation mode');
                }
            } else {
                throw new Error(data.message || 'Failed to toggle V2 mode');
            }
            
        } catch (error) {
            console.error('Failed to toggle V2 mode:', error);
            this.showToast('error', `Failed to toggle V2 mode: ${error.message}`);
            
            
            document.getElementById('v2ModeToggle').checked = this.v2ModeEnabled;
        } finally {
            this.hideLoading();
        }
    }

    async reconnectSession() {
        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiUrl}/api/session/reconnect`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            
            if (data.success) {
                const ageMinutes = Math.floor(data.session_age_seconds / 60);
                const ageSeconds = Math.floor(data.session_age_seconds % 60);
                
                this.showToast('success', `Reconnecting with saved ${data.mode} session`);
                this.addConsoleMessage('success', `Reconnection initiated with saved ${data.mode} session`);
                this.addConsoleMessage('info', `Session age: ${ageMinutes}m ${ageSeconds}s | Handle: ${data.handle_preview}`);
                
                if (data.session_age_seconds > 1800) {
                    this.addConsoleMessage('warning', 'Session is older than 30 minutes - reconnection may fail');
                }
            } else {
                throw new Error(data.message || 'Failed to reconnect session');
            }
            
        } catch (error) {
            console.error('Failed to reconnect session:', error);
            
            if (error.message && error.message.includes('404')) {
                this.showToast('error', 'No saved session handle found');
                this.addConsoleMessage('error', 'No saved session available. Start a new session first.');
            } else if (error.message && error.message.includes('409')) {
                this.showToast('warning', 'A session is already active');
                this.addConsoleMessage('warning', 'Cannot reconnect - an active session already exists');
            } else {
                this.showToast('error', `Failed to reconnect: ${error.message}`);
                this.addConsoleMessage('error', `Reconnection failed: ${error.message}`);
            }
        } finally {
            this.hideLoading();
        }
    }

    async freshStart() {
        if (!confirm('Are you sure you want to clear the session and restart fresh? This will clear the saved session handle.')) {
            return;
        }
        
        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiUrl}/api/session/fresh-start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            
            if (data.success) {
                this.showToast('success', 'Session cleared - restarting fresh');
                this.addConsoleMessage('success', 'Saved session handle cleared');
                this.addConsoleMessage('info', 'AI will restart with a completely fresh session');
            } else {
                throw new Error(data.message || 'Failed to clear session');
            }
            
        } catch (error) {
            console.error('Failed to start fresh:', error);
            this.showToast('error', `Failed to start fresh: ${error.message}`);
            this.addConsoleMessage('error', `Fresh start failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) {
            this.showToast('warning', 'Please enter a message');
            return;
        }

        const messageType = document.querySelector('input[name="messageType"]:checked').value;
        const isSystemInstruction = messageType === 'system';

        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiUrl}/api/chat/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    system_instruction: isSystemInstruction,
                    turn_complete: true
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.showToast('success', 'Message sent successfully');
                this.addConsoleMessage('success', `Sent ${isSystemInstruction ? 'system instruction' : 'message'}: ${message}`);
                messageInput.value = '';
                document.getElementById('charCounter').textContent = '0/1000';
            } else {
                throw new Error(data.message || 'Failed to send message');
            }
            
        } catch (error) {
            console.error('Failed to send message:', error);
            this.showToast('error', `Failed to send message: ${error.message}`);
            this.addConsoleMessage('error', `Failed to send message: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async sendQuickMessage(message) {
        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiUrl}/api/chat/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    system_instruction: true,
                    turn_complete: true
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.showToast('success', 'Quick instruction sent');
                this.addConsoleMessage('success', `Quick instruction: ${message}`);
            } else {
                throw new Error(data.message || 'Failed to send instruction');
            }
            
        } catch (error) {
            console.error('Failed to send quick message:', error);
            this.showToast('error', `Failed to send instruction: ${error.message}`);
            this.addConsoleMessage('error', `Failed to send instruction: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    addConsoleMessage(type, message, newLine = true) {
        const timestamp = new Date().toLocaleTimeString();
        
        
        if (!newLine && type === 'response') {
            const lastLine = this.consoleOutput.lastElementChild;
            if (lastLine && lastLine.classList.contains('console-line') && lastLine.classList.contains('response')) {
                const messageSpan = lastLine.querySelector('.message');
                if (messageSpan) {
                    messageSpan.textContent += message.replace('AI Responce: ', '');
                    
                    if (this.autoScroll) {
                        this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
                    }
                    return;
                }
            }
        }
        
        
        const consoleLine = document.createElement('div');
        consoleLine.className = `console-line ${type}`;
        
        consoleLine.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="message">${this.escapeHtml(message)}</span>
        `;
        
        this.consoleOutput.appendChild(consoleLine);
        
        if (this.autoScroll) {
            this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        }

        
        const lines = this.consoleOutput.children;
        if (lines.length > 1000) {
            this.consoleOutput.removeChild(lines[0]);
        }
    }

    addFunctionCallMessage(data) {
        const timestamp = new Date().toLocaleTimeString();
        const { name, args } = data;
        
        const consoleLine = document.createElement('div');
        consoleLine.className = 'console-line function-call';
        
        const argsJson = JSON.stringify(args, null, 2);
        const argsId = `args-${Date.now()}-${Math.random()}`;
        
        consoleLine.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="message">
                <strong>⚡ Function Call:</strong> <code>${this.escapeHtml(name)}</code>
                <button class="toggle-details" onclick="document.getElementById('${argsId}').classList.toggle('hidden')">
                    [Show Args]
                </button>
                <pre id="${argsId}" class="function-details hidden">${this.escapeHtml(argsJson)}</pre>
            </span>
        `;
        
        this.consoleOutput.appendChild(consoleLine);
        
        if (this.autoScroll) {
            this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        }

        const lines = this.consoleOutput.children;
        if (lines.length > 1000) {
            this.consoleOutput.removeChild(lines[0]);
        }
    }

    addFunctionResponseMessage(data) {
        const timestamp = new Date().toLocaleTimeString();
        const { name, response } = data;
        
        const consoleLine = document.createElement('div');
        consoleLine.className = 'console-line function-response';
        
        const responseJson = JSON.stringify(response, null, 2);
        const responseId = `response-${Date.now()}-${Math.random()}`;
        
        const success = response?.success !== false;
        const statusIcon = success ? '✓' : '✗';
        const message = response?.message || 'No message';
        
        consoleLine.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="message">
                <strong>${statusIcon} Function Response:</strong> <code>${this.escapeHtml(name)}</code>
                <span class="response-summary">${this.escapeHtml(message)}</span>
                <button class="toggle-details" onclick="document.getElementById('${responseId}').classList.toggle('hidden')">
                    [Show Full Response]
                </button>
                <pre id="${responseId}" class="function-details hidden">${this.escapeHtml(responseJson)}</pre>
            </span>
        `;
        
        this.consoleOutput.appendChild(consoleLine);
        
        if (this.autoScroll) {
            this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        }

        const lines = this.consoleOutput.children;
        if (lines.length > 1000) {
            this.consoleOutput.removeChild(lines[0]);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearConsole() {
        this.consoleOutput.innerHTML = '';
        this.addConsoleMessage('system', 'Console cleared');
    }

    toggleAutoScroll() {
        this.autoScroll = !this.autoScroll;
        const button = document.getElementById('toggleAutoScroll');
        const icon = button.querySelector('i');
        
        if (this.autoScroll) {
            button.innerHTML = '<i class="fas fa-arrow-down"></i> Auto-scroll: ON';
            icon.className = 'fas fa-arrow-down';
        } else {
            button.innerHTML = '<i class="fas fa-pause"></i> Auto-scroll: OFF';
            icon.className = 'fas fa-pause';
        }
    }

    toggleLegend() {
        const legend = document.getElementById('messageLegend');
        const button = document.getElementById('showLegend');
        
        legend.classList.toggle('hidden');
        
        if (legend.classList.contains('hidden')) {
            button.innerHTML = '<i class="fas fa-info-circle"></i> Legend';
        } else {
            button.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Legend';
        }
    }

    showLoading() {
        document.getElementById('loadingOverlay').classList.remove('hidden');
        document.getElementById('sendButton').disabled = true;
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.add('hidden');
        document.getElementById('sendButton').disabled = false;
    }

    showToast(type, message) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${this.escapeHtml(message)}</span>
            </div>
        `;
        
        document.getElementById('toastContainer').appendChild(toast);
        
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }

    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return;
        }

        this.updateWebSocketStatus('connecting');

        try {
            this.websocket = new WebSocket(this.wsUrl);
            
            this.websocket.onopen = (event) => {
                this.addConsoleMessage('system', 'WebSocket connected - Real-time monitoring active');
                this.updateWebSocketStatus('connected');
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.websocket.onclose = (event) => {
                this.addConsoleMessage('warning', 'WebSocket disconnected - Attempting to reconnect...');
                this.updateWebSocketStatus('disconnected');
                this.scheduleReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.addConsoleMessage('error', 'WebSocket connection error');
                this.updateWebSocketStatus('error');
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.addConsoleMessage('error', 'Failed to establish WebSocket connection');
            this.updateWebSocketStatus('error');
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.addConsoleMessage('error', 'Max reconnection attempts reached. Please refresh the page.');
            this.updateWebSocketStatus('failed');
            return;
        }

        setTimeout(() => {
            this.reconnectAttempts++;
            this.addConsoleMessage('system', `Reconnecting... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            this.connectWebSocket();
        }, this.reconnectDelay);

        
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
    }

    handleWebSocketMessage(message) {
        const { type, data, timestamp } = message;
        
        
        if (type !== 'user_input_transcription' && type !== 'heartbeat' && type !== 'pong') {
            this.flushUserSpeechBuffer();
        }
        
        switch (type) {
            case 'connection':
                this.addConsoleMessage('success', data.message);
                break;
                
            case 'user_message':
                this.addConsoleMessage('info', data.message);
                break;
                
            case 'system_instruction':
                this.addConsoleMessage('system', data.message);
                break;
                
            case 'function_call':
                this.addFunctionCallMessage(data);
                break;
                
            case 'function_response':
                this.handleFunctionResponse(data);
                break;
                
            case 'user_input_transcription':
                
                this.userSpeechBuffer.push({
                    text: data.text,
                    timestamp: timestamp
                });
                
                
                if (this.userSpeechBufferTimeout) {
                    clearTimeout(this.userSpeechBufferTimeout);
                }
                
                this.userSpeechBufferTimeout = setTimeout(() => {
                    this.flushUserSpeechBuffer();
                }, this.userSpeechGroupingDelay);
                break;
                
            case 'text_chunk':
                
                this.addConsoleMessage('response', `AI Response: ${data.text}`, false);
                
                
                this.checkForFunctionResponse(data.text);
                break;
                
            case 'transcription_chunk':
                
                break;
                
            case 'complete_response':
                
                this.addConsoleMessage('response', `AI Response (complete): ${data.text}`);
                
                
                this.checkForFunctionResponse(data.text);
                break;
                
            case 'system':
                this.addConsoleMessage('system', data.message);
                
                
                this.checkForFunctionResponse(data.message);
                
                
                if (data.message && data.message.includes('Personality switched to:')) {
                    
                    setTimeout(() => this.loadPersonalities(), 1000);
                }
                break;
                
            case 'error':
                this.addConsoleMessage('error', data.message);
                break;
                
            case 'heartbeat':
                
                break;
                
            case 'pong':
                
                break;
                
            default:
                console.log('Unknown WebSocket message type:', type);
        }
    }

    handleFunctionResponse(data) {
        this.addFunctionResponseMessage(data);
        this.updateYapModeFromResponse(data.response || data);
    }

    checkForFunctionResponse(message) {
        
        
        const functionResponseMatch = message.match(/Function response:\s*({.*})/);
        
        if (functionResponseMatch) {
            try {
                let jsonString = functionResponseMatch[1];
                
                
                jsonString = jsonString
                    .replace(/'/g, '"')          
                    .replace(/True/g, 'true')    
                    .replace(/False/g, 'false')  
                    .replace(/None/g, 'null');   
                
                const responseData = JSON.parse(jsonString);
                this.updateYapModeFromResponse(responseData);
            } catch (error) {
                console.error('Failed to parse function response JSON:', error);
                console.log('Raw match:', functionResponseMatch[1]);
            }
        }
    }

    updateYapModeFromResponse(responseData) {
        
        if (responseData && typeof responseData.yap_mode_enabled === 'boolean') {
            const newYapModeState = responseData.yap_mode_enabled;
            const toggle = document.getElementById('yapModeToggle');
            
            
            if (toggle.checked !== newYapModeState) {
                toggle.checked = newYapModeState;
                this.yapModeEnabled = newYapModeState;
                
                
                const statusMessage = newYapModeState ? 'enabled' : 'disabled';
                this.showToast('info', `Yap mode automatically ${statusMessage}`);
                this.addConsoleMessage('success', `Yap mode toggle updated: ${statusMessage}`);
                
                
                if (responseData.message) {
                    this.addConsoleMessage('info', `Response: ${responseData.message}`);
                }
            }
        }
    }

    flushUserSpeechBuffer() {
        if (this.userSpeechBuffer.length === 0) {
            return;
        }
        
        
        if (this.userSpeechBufferTimeout) {
            clearTimeout(this.userSpeechBufferTimeout);
            this.userSpeechBufferTimeout = null;
        }
        
        
        const combinedText = this.userSpeechBuffer.map(item => item.text).join(' ').trim();
        
        
        if (combinedText) {
            this.addConsoleMessage('user-speech', `[User Speech]: ${combinedText}`);
        }
        
        
        this.userSpeechBuffer = [];
    }

    updateWebSocketStatus(status) {
        const wsStatus = document.getElementById('wsStatus');
        
        switch (status) {
            case 'connected':
                wsStatus.className = 'api-status-badge online';
                wsStatus.textContent = 'Connected';
                break;
            case 'connecting':
                wsStatus.className = 'api-status-badge';
                wsStatus.textContent = 'Connecting...';
                break;
            case 'disconnected':
                wsStatus.className = 'api-status-badge offline';
                wsStatus.textContent = 'Disconnected';
                break;
            case 'error':
                wsStatus.className = 'api-status-badge offline';
                wsStatus.textContent = 'Error';
                break;
            case 'failed':
                wsStatus.className = 'api-status-badge offline';
                wsStatus.textContent = 'Failed';
                break;
            default:
                wsStatus.className = 'api-status-badge';
                wsStatus.textContent = 'Unknown';
        }
    }

    sendWebSocketPing() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'ping',
                timestamp: Date.now()
            }));
        }
    }


    simulateAIResponse(text) {
        this.addConsoleMessage('response', `AI Response: ${text}`);
    }

    
    disconnect() {
        
        this.flushUserSpeechBuffer();
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
    
    
    async loadPersonalities() {
        try {
            this.showPersonalitiesLoading(true);
            
            const response = await fetch(`${this.apiUrl}/api/personalities`);
            const data = await response.json();
            
            if (data.success) {
                this.personalities = data.personalities;
                this.currentPersonality = data.current;
                this.renderPersonalities();
                this.updateCurrentPersonalityDisplay();
                
                this.addConsoleMessage('system', `Loaded ${data.count} personalities`);
            } else {
                throw new Error(data.message || 'Failed to load personalities');
            }
            
        } catch (error) {
            console.error('Failed to load personalities:', error);
            this.showToast('error', `Failed to load personalities: ${error.message}`);
            this.addConsoleMessage('error', `Failed to load personalities: ${error.message}`);
            this.showPersonalitiesError();
        } finally {
            this.showPersonalitiesLoading(false);
        }
    }
    
    async switchPersonality(personalityId) {
        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiUrl}/api/personalities/switch/${personalityId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            
            if (data.success) {
                this.currentPersonality = personalityId;
                this.renderPersonalities();
                this.updateCurrentPersonalityDisplay();
                
                this.showToast('success', `Switched to ${data.personality.name} personality`);
                this.addConsoleMessage('success', `Personality switched to: ${data.personality.name}`);
            } else {
                throw new Error(data.message || 'Failed to switch personality');
            }
            
        } catch (error) {
            console.error('Failed to switch personality:', error);
            this.showToast('error', `Failed to switch personality: ${error.message}`);
            this.addConsoleMessage('error', `Failed to switch personality: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    renderPersonalities() {
        const personalitiesList = document.getElementById('personalitiesList');
        
        if (!this.personalities || this.personalities.length === 0) {
            personalitiesList.innerHTML = `
                <div class="personalities-loading">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>No personalities available</span>
                </div>
            `;
            return;
        }
        
        const personalitiesHtml = this.personalities.map(personality => {
            const isActive = personality.id === this.currentPersonality;
            const isDisabled = !personality.enabled;
            
            let statusText = '';
            let statusClass = '';
            
            if (isActive) {
                statusText = 'Active';
                statusClass = 'active';
            } else if (isDisabled) {
                statusText = 'Disabled';
                statusClass = 'disabled';
            } else {
                statusText = 'Available';
                statusClass = '';
            }
            
            return `
                <button class="personality-button ${isActive ? 'active' : ''} ${isDisabled ? 'disabled' : ''}"
                        data-personality-id="${personality.id}"
                        ${isDisabled ? 'disabled' : ''}>
                    <div class="personality-name">${this.escapeHtml(personality.name)}</div>
                    <div class="personality-description">${this.escapeHtml(personality.description)}</div>
                    <div class="personality-status ${statusClass}">
                        <i class="fas fa-${isActive ? 'check-circle' : isDisabled ? 'ban' : 'circle'}"></i>
                        ${statusText}
                    </div>
                </button>
            `;
        }).join('');
        
        personalitiesList.innerHTML = personalitiesHtml;
        
        
        personalitiesList.querySelectorAll('.personality-button:not(.disabled)').forEach(button => {
            button.addEventListener('click', (e) => {
                const personalityId = e.currentTarget.dataset.personalityId;
                if (personalityId && personalityId !== this.currentPersonality) {
                    this.switchPersonality(personalityId);
                }
            });
        });
    }
    
    updateCurrentPersonalityDisplay() {
        const currentPersonalityName = document.getElementById('currentPersonalityName');
        
        if (this.currentPersonality) {
            const current = this.personalities.find(p => p.id === this.currentPersonality);
            if (current) {
                currentPersonalityName.textContent = current.name;
                currentPersonalityName.style.color = 'var(--accent-primary)';
            } else {
                currentPersonalityName.textContent = this.currentPersonality;
                currentPersonalityName.style.color = 'var(--accent-warning)';
            }
        } else {
            currentPersonalityName.textContent = 'None';
            currentPersonalityName.style.color = 'var(--text-muted)';
        }
    }
    
    showPersonalitiesLoading(show) {
        const personalitiesList = document.getElementById('personalitiesList');
        
        if (show) {
            personalitiesList.innerHTML = `
                <div class="personalities-loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>Loading personalities...</span>
                </div>
            `;
        }
    }
    
    showPersonalitiesError() {
        const personalitiesList = document.getElementById('personalitiesList');
        personalitiesList.innerHTML = `
            <div class="personalities-loading">
                <i class="fas fa-exclamation-triangle"></i>
                <span>Failed to load personalities</span>
                <button class="btn btn-secondary btn-small" onclick="window.gabrielPanel.loadPersonalities()" style="margin-top: 10px;">
                    <i class="fas fa-retry"></i> Retry
                </button>
            </div>
        `;
    }
    
    
    async loadVRChatControlsStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/api/vrchat/controls/status`);
            const data = await response.json();
            
            if (data.success) {
                this.updateVRChatStatus(data.controls);
            } else {
                throw new Error(data.message || 'Failed to get VRChat controls status');
            }
            
        } catch (error) {
            console.error('Failed to load VRChat controls status:', error);
            this.updateVRChatStatus({
                enabled: false,
                connected: false,
                safe_mode_enabled: false,
                voice_enabled: true
            });
        }
    }
    
    updateVRChatStatus(controls) {
        const vrchatStatus = document.getElementById('vrchatStatus');
        const voiceToggle = document.getElementById('voiceToggle');
        
        if (controls.enabled && controls.connected) {
            vrchatStatus.className = 'api-status-badge online';
            vrchatStatus.textContent = 'Connected';
            
            
            voiceToggle.checked = controls.voice_enabled;
        } else if (controls.enabled && !controls.connected) {
            vrchatStatus.className = 'api-status-badge offline';
            vrchatStatus.textContent = 'Disconnected';
        } else {
            vrchatStatus.className = 'api-status-badge offline';
            vrchatStatus.textContent = 'Disabled';
        }
    }
    
    async enableSafeMode() {
        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiUrl}/api/vrchat/controls/safe-mode`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            
            if (data.success) {
                this.showToast('success', 'VRChat Safe Mode enabled');
                this.addConsoleMessage('success', 'VRChat Safe Mode enabled');
                
                
                setTimeout(() => this.loadVRChatControlsStatus(), 1000);
            } else {
                throw new Error(data.message || 'Failed to enable Safe Mode');
            }
            
        } catch (error) {
            console.error('Failed to enable Safe Mode:', error);
            this.showToast('error', `Failed to enable Safe Mode: ${error.message}`);
            this.addConsoleMessage('error', `Failed to enable Safe Mode: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async toggleVRChatVoice(enable) {
        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiUrl}/api/vrchat/controls/voice/toggle`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    enable: enable
                })
            });

            const data = await response.json();
            
            if (data.success) {
                const action = data.voice_enabled ? 'enabled' : 'disabled';
                this.showToast('success', `VRChat voice ${action}`);
                this.addConsoleMessage('success', `VRChat voice ${action}`);
                
                
                document.getElementById('voiceToggle').checked = data.voice_enabled;
                
                
                setTimeout(() => this.loadVRChatControlsStatus(), 1000);
            } else {
                throw new Error(data.message || 'Failed to toggle voice');
            }
            
        } catch (error) {
            console.error('Failed to toggle VRChat voice:', error);
            this.showToast('error', `Failed to toggle voice: ${error.message}`);
            this.addConsoleMessage('error', `Failed to toggle voice: ${error.message}`);
            
            
            document.getElementById('voiceToggle').checked = !enable;
        } finally {
            this.hideLoading();
        }
    }
}


document.addEventListener('DOMContentLoaded', () => {
    window.gabrielPanel = new GabrielControlPanel();
    
    
    setInterval(() => {
        if (window.gabrielPanel) {
            window.gabrielPanel.sendWebSocketPing();
        }
    }, 30000);
});


window.addEventListener('beforeunload', () => {
    if (window.gabrielPanel) {
        window.gabrielPanel.disconnect();
    }
});


document.addEventListener('keydown', (e) => {
    
    if (e.ctrlKey && e.key === 'Enter') {
        const messageInput = document.getElementById('messageInput');
        if (document.activeElement === messageInput) {
            e.preventDefault();
            document.getElementById('sendButton').click();
        }
    }
    
    
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        window.gabrielPanel.clearConsole();
    }
});