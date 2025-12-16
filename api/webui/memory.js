

let currentMemories = [];
let filteredMemories = [];
let currentSort = { field: 'updated_at', direction: 'desc' };
let currentPage = 1;
let itemsPerPage = 20;
let isEditMode = false;
let editingKey = null;
let isAnyModalOpen = false;
let lastViewMemoryTime = 0;
let isViewingMemory = false;

const API_BASE = (() => {
    const currentHost = window.location.host;
    const currentPort = window.location.port;
    const currentHostname = window.location.hostname;
    
    let apiBase;
    
    if (currentPort === '5069' || currentPort === '5500' || 
        currentHost.includes('5069') || currentHost.includes('5500')) {
        apiBase = `http://${currentHostname}:8000`;
        console.log(`Memory Manager: Running on WebUI server (port ${currentPort}), using Chat API at ${apiBase}`);
    }
    
    else if (currentPort === '8000') {
        apiBase = window.location.origin;
        console.log(`Memory Manager: Running on Chat API server, using same origin: ${apiBase}`);
    }
    
    else {
        apiBase = `http://${currentHostname}:8000`;
        console.log(`Memory Manager: Unknown port (${currentPort}), defaulting to Chat API at ${apiBase}`);
    }
    
    return apiBase;
})();


let elements = {};


document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    initializeEventListeners();
    
    
    showToast(`Connecting to API at ${API_BASE}...`, 'info', 3000);
    
    
    if (elements.apiUrl) {
        elements.apiUrl.textContent = API_BASE;
    }
    
    
    testConnection();
});


async function testConnection() {
    try {
        
        await checkApiStatus();
        
        
        await Promise.all([
            loadMemories(), 
            loadMemoryStatsWithRetry()
        ]);
        
        
        if (elements.connectionErrorState) {
            elements.connectionErrorState.classList.add('hidden');
        }
        
    } catch (error) {
        console.error('Connection test failed:', error);
        showConnectionError();
    }
}


async function loadMemoryStatsWithRetry(maxRetries = 3, delay = 1000) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            await loadMemoryStats();
            return;
        } catch (error) {
            console.warn(`Memory stats load attempt ${attempt}/${maxRetries} failed:`, error.message);
            
            if (attempt === maxRetries) {
                
                console.error('All memory stats load attempts failed');
                showToast(`Failed to load memory statistics after ${maxRetries} attempts. Check if the API is running on port 8000.`, 'error', 15000);
                throw error;
            }
            
            
            await new Promise(resolve => setTimeout(resolve, delay));
            delay *= 1.5;
        }
    }
}


function showConnectionError() {
    
    hideLoading();
    elements.emptyState.classList.add('hidden');
    elements.memoryTableContainer.classList.add('hidden');
    
    
    elements.connectionErrorState.classList.remove('hidden');
    
    
    if (elements.apiUrl) {
        elements.apiUrl.textContent = API_BASE;
    }
}


function showConnectionHelp() {
    const helpMessage = `
        <strong>Unable to connect to API</strong><br><br>
        Please ensure:<br>
        • API server is running<br>
        • API is accessible at ${API_BASE}<br>
        • Memory endpoints are properly configured<br><br>
        <em>If running locally, start the API server on port 8000</em>
    `;
    
    showToast(helpMessage, 'error', 15000);
}


function initializeElements() {
    elements = {
        
        statusDot: document.getElementById('statusDot'),
        statusText: document.getElementById('statusText'),
        
        
        addMemoryBtn: document.getElementById('addMemoryBtn'),
        searchInput: document.getElementById('searchInput'),
        clearSearchBtn: document.getElementById('clearSearchBtn'),
        memoryTypeFilter: document.getElementById('memoryTypeFilter'),
        categoryFilter: document.getElementById('categoryFilter'),
        refreshBtn: document.getElementById('refreshBtn'),
        
        
        totalMemories: document.getElementById('totalMemories'),
        longTermCount: document.getElementById('longTermCount'),
        shortTermCount: document.getElementById('shortTermCount'),
        quickNoteCount: document.getElementById('quickNoteCount'),
        
        
        memoryCount: document.getElementById('memoryCount'),
        loadingState: document.getElementById('loadingState'),
        connectionErrorState: document.getElementById('connectionErrorState'),
        emptyState: document.getElementById('emptyState'),
        memoryTableContainer: document.getElementById('memoryTableContainer'),
        memoryTable: document.getElementById('memoryTable'),
        memoryTableBody: document.getElementById('memoryTableBody'),
        apiUrl: document.getElementById('apiUrl'),
        
        
        paginationContainer: document.getElementById('paginationContainer'),
        paginationInfo: document.getElementById('paginationInfo'),
        prevPageBtn: document.getElementById('prevPageBtn'),
        nextPageBtn: document.getElementById('nextPageBtn'),
        pageInfo: document.getElementById('pageInfo'),
        
        
        memoryModal: document.getElementById('memoryModal'),
        modalTitle: document.getElementById('modalTitle'),
        memoryForm: document.getElementById('memoryForm'),
        memoryKey: document.getElementById('memoryKey'),
        memoryContent: document.getElementById('memoryContent'),
        memoryCategory: document.getElementById('memoryCategory'),
        memoryType: document.getElementById('memoryType'),
        memoryTags: document.getElementById('memoryTags'),
        saveMemoryBtn: document.getElementById('saveMemoryBtn'),
        
        
        deleteModal: document.getElementById('deleteModal'),
        deleteMemoryKey: document.getElementById('deleteMemoryKey'),
        deleteMemoryCategory: document.getElementById('deleteMemoryCategory'),
        deleteMemoryType: document.getElementById('deleteMemoryType'),
        confirmDeleteBtn: document.getElementById('confirmDeleteBtn'),
        
        
        viewModal: document.getElementById('viewModal'),
        viewMemoryKey: document.getElementById('viewMemoryKey'),
        viewMemoryContent: document.getElementById('viewMemoryContent'),
        viewMemoryCategory: document.getElementById('viewMemoryCategory'),
        viewMemoryType: document.getElementById('viewMemoryType'),
        viewMemoryCreated: document.getElementById('viewMemoryCreated'),
        viewMemoryUpdated: document.getElementById('viewMemoryUpdated'),
        viewMemoryAccessCount: document.getElementById('viewMemoryAccessCount'),
        viewMemoryTags: document.getElementById('viewMemoryTags'),
        editFromViewBtn: document.getElementById('editFromViewBtn'),
        
        
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingText: document.getElementById('loadingText'),
        toastContainer: document.getElementById('toastContainer')
    };
}


function initializeEventListeners() {
    
    elements.addMemoryBtn.addEventListener('click', openAddMemoryModal);
    elements.refreshBtn.addEventListener('click', refreshData);
    
    
    elements.searchInput.addEventListener('input', debounce(handleSearch, 300));
    elements.clearSearchBtn.addEventListener('click', clearSearch);
    elements.memoryTypeFilter.addEventListener('change', handleFilterChange);
    elements.categoryFilter.addEventListener('change', handleFilterChange);
    
    
    const sortableHeaders = elements.memoryTable.querySelectorAll('th.sortable');
    sortableHeaders.forEach(header => {
        header.addEventListener('click', () => handleSort(header.dataset.sort));
    });
    
    
    elements.prevPageBtn.addEventListener('click', () => changePage(currentPage - 1));
    elements.nextPageBtn.addEventListener('click', () => changePage(currentPage + 1));
    
    
    elements.memoryForm.addEventListener('submit', handleFormSubmit);
    
    
    elements.editFromViewBtn.addEventListener('click', editFromView);
    elements.confirmDeleteBtn.addEventListener('click', confirmDelete);
    
    
    [elements.memoryModal, elements.deleteModal, elements.viewModal].forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeAllModals();
            }
        });
    });
    
    
    document.addEventListener('keydown', handleKeyboardShortcuts);
}


async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/chat/status`);
        
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('API server is not responding with JSON. Make sure the API is running on the correct port.');
        }
        
        const data = await response.json();
        
        if (data.success) {
            updateStatus('connected', 'Connected');
        } else {
            updateStatus('error', 'API Error');
        }
    } catch (error) {
        console.error('Failed to check API status:', error);
        updateStatus('error', `API Unavailable: ${error.message}`);
        showToast(`API connection failed: ${error.message}. Make sure the API is running on port 8000.`, 'error', 10000);
    }
}


function updateStatus(status, text) {
    elements.statusDot.className = `status-dot ${status}`;
    elements.statusText.textContent = text;
}


async function loadMemories() {
    
    if (isAnyModalOpen || isViewingMemory) {
        console.log('Skipping memory refresh - modal state flags active (isAnyModalOpen =', isAnyModalOpen, ', isViewingMemory =', isViewingMemory, ')');
        return;
    }
    
    
    const timeSinceLastView = Date.now() - lastViewMemoryTime;
    if (timeSinceLastView < 15000) {
        console.log('Skipping memory refresh - recently viewed memory (', timeSinceLastView, 'ms ago)');
        return;
    }
    
    
    const modalsVisible = [
        !elements.memoryModal.classList.contains('hidden'),
        !elements.viewModal.classList.contains('hidden'), 
        !elements.deleteModal.classList.contains('hidden'),
        
        document.querySelector('.modal:not(.hidden)') !== null
    ];
    
    if (modalsVisible.some(visible => visible)) {
        console.log('Skipping memory refresh - modal is visible in DOM');
        return;
    }
    
    showLoading();
    
    try {
        const params = new URLSearchParams();
        
        if (elements.memoryTypeFilter.value) {
            params.append('memory_type', elements.memoryTypeFilter.value);
        }
        
        if (elements.categoryFilter.value) {
            params.append('category', elements.categoryFilter.value);
        }
        
        params.append('limit', '1000');
        
        const response = await fetch(`${API_BASE}/api/memory/list?${params}`);
        
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Server returned HTML instead of JSON. Check if the API is running and memory endpoints are available.');
        }
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            currentMemories = data.memories;
            updateCategoryFilter();
            applyFiltersAndSearch();
            showToast('Memories loaded successfully', 'success');
        } else {
            throw new Error(data.message || 'Failed to load memories');
        }
    } catch (error) {
        console.error('Failed to load memories:', error);
        showToast('Failed to load memories: ' + error.message, 'error', 10000);
        currentMemories = [];
        applyFiltersAndSearch();
    } finally {
        hideLoading();
    }
}


async function loadMemoryStats() {
    
    if (isAnyModalOpen || isViewingMemory) {
        console.log('Skipping memory stats refresh - modal state flags active (isAnyModalOpen =', isAnyModalOpen, ', isViewingMemory =', isViewingMemory, ')');
        return;
    }
    
    
    const timeSinceLastView = Date.now() - lastViewMemoryTime;
    if (timeSinceLastView < 15000) {
        console.log('Skipping memory stats refresh - recently viewed memory (', timeSinceLastView, 'ms ago)');
        return;
    }
    
    
    const modalsVisible = [
        !elements.memoryModal.classList.contains('hidden'),
        !elements.viewModal.classList.contains('hidden'), 
        !elements.deleteModal.classList.contains('hidden'),
        
        document.querySelector('.modal:not(.hidden)') !== null
    ];
    
    if (modalsVisible.some(visible => visible)) {
        console.log('Skipping memory stats refresh - modal is visible in DOM');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/memory/stats`);
        
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error(`Server returned ${response.status} with non-JSON response. Content-Type: ${contentType || 'none'}`);
        }
        
        let data;
        try {
            data = await response.json();
        } catch (parseError) {
            throw new Error(`Failed to parse JSON response: ${parseError.message}`);
        }
        
        if (!response.ok) {
            const errorDetail = data.detail || data.message || 'Unknown error';
            throw new Error(`HTTP ${response.status}: ${errorDetail}`);
        }
        
        if (data.success) {
            const stats = data.stats;
            elements.totalMemories.textContent = stats.total || 0;
            elements.longTermCount.textContent = stats.long_term || 0;
            elements.shortTermCount.textContent = stats.short_term || 0;
            elements.quickNoteCount.textContent = stats.quick_note || 0;
        } else {
            throw new Error(data.message || 'Failed to load stats');
        }
    } catch (error) {
        console.error('Failed to load memory stats:', error);
        console.error('API_BASE:', API_BASE);
        console.error('Full error details:', {
            message: error.message,
            stack: error.stack,
            url: `${API_BASE}/api/memory/stats`
        });
        
        
        elements.totalMemories.textContent = '-';
        elements.longTermCount.textContent = '-';
        elements.shortTermCount.textContent = '-';
        elements.quickNoteCount.textContent = '-';
        
        
        if (!error.message.includes('HTML instead of JSON') && !error.message.includes('Failed to fetch')) {
            showToast(`Failed to load memory statistics: ${error.message}`, 'error', 10000);
        }
    }
}


function updateCategoryFilter() {
    const categories = [...new Set(currentMemories.map(m => m.category))].sort();
    const currentValue = elements.categoryFilter.value;
    
    
    elements.categoryFilter.innerHTML = '<option value="">All Categories</option>';
    
    
    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        elements.categoryFilter.appendChild(option);
    });
    
    
    if (categories.includes(currentValue)) {
        elements.categoryFilter.value = currentValue;
    }
}


function applyFiltersAndSearch() {
    let filtered = [...currentMemories];
    
    
    const searchTerm = elements.searchInput.value.trim().toLowerCase();
    if (searchTerm) {
        filtered = filtered.filter(memory => {
            const keyMatch = memory.key.toLowerCase().includes(searchTerm);
            const contentMatch = memory.content && memory.content.toLowerCase().includes(searchTerm);
            const categoryMatch = memory.category.toLowerCase().includes(searchTerm);
            const tagsMatch = Array.isArray(memory.tags) && memory.tags.some(tag => 
                tag.toLowerCase().includes(searchTerm)
            );
            return keyMatch || contentMatch || categoryMatch || tagsMatch;
        });
    }
    
    
    const memoryType = elements.memoryTypeFilter.value;
    if (memoryType) {
        filtered = filtered.filter(memory => memory.memory_type === memoryType);
    }
    
    
    const category = elements.categoryFilter.value;
    if (category) {
        filtered = filtered.filter(memory => memory.category === category);
    }
    
    filteredMemories = filtered;
    applySorting();
    updatePagination();
    renderMemoryTable();
}


function applySorting() {
    filteredMemories.sort((a, b) => {
        let aVal = a[currentSort.field];
        let bVal = b[currentSort.field];
        
        
        if (currentSort.field === 'access_count') {
            aVal = parseInt(aVal) || 0;
            bVal = parseInt(bVal) || 0;
        } else if (currentSort.field.includes('_at')) {
            aVal = new Date(aVal);
            bVal = new Date(bVal);
        } else {
            aVal = String(aVal).toLowerCase();
            bVal = String(bVal).toLowerCase();
        }
        
        let comparison = 0;
        if (aVal > bVal) comparison = 1;
        if (aVal < bVal) comparison = -1;
        
        return currentSort.direction === 'desc' ? -comparison : comparison;
    });
}


function handleSearch() {
    currentPage = 1;
    applyFiltersAndSearch();
}


function clearSearch() {
    elements.searchInput.value = '';
    handleSearch();
}


function handleFilterChange() {
    currentPage = 1;
    applyFiltersAndSearch();
}


function handleSort(field) {
    if (currentSort.field === field) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.field = field;
        currentSort.direction = 'asc';
    }
    
    applySorting();
    renderMemoryTable();
    updateSortIndicators();
}


function updateSortIndicators() {
    const headers = elements.memoryTable.querySelectorAll('th.sortable');
    headers.forEach(header => {
        const icon = header.querySelector('i');
        const field = header.dataset.sort;
        
        if (field === currentSort.field) {
            icon.className = currentSort.direction === 'asc' ? 'fas fa-sort-up' : 'fas fa-sort-down';
        } else {
            icon.className = 'fas fa-sort';
        }
    });
}


function updatePagination() {
    const totalItems = filteredMemories.length;
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    
    
    if (currentPage > totalPages && totalPages > 0) {
        currentPage = totalPages;
    } else if (currentPage < 1) {
        currentPage = 1;
    }
    
    
    const startItem = totalItems === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1;
    const endItem = Math.min(currentPage * itemsPerPage, totalItems);
    
    elements.paginationInfo.textContent = `Showing ${startItem}-${endItem} of ${totalItems} memories`;
    elements.pageInfo.textContent = totalPages > 0 ? `Page ${currentPage} of ${totalPages}` : 'Page 0 of 0';
    
    
    elements.prevPageBtn.disabled = currentPage <= 1;
    elements.nextPageBtn.disabled = currentPage >= totalPages;
    
    
    if (totalPages <= 1) {
        elements.paginationContainer.classList.add('hidden');
    } else {
        elements.paginationContainer.classList.remove('hidden');
    }
}


function changePage(page) {
    const totalPages = Math.ceil(filteredMemories.length / itemsPerPage);
    if (page >= 1 && page <= totalPages) {
        currentPage = page;
        renderMemoryTable();
        updatePagination();
    }
}


function renderMemoryTable() {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const pageMemories = filteredMemories.slice(startIndex, endIndex);
    
    
    elements.memoryCount.textContent = `${filteredMemories.length} ${filteredMemories.length === 1 ? 'memory' : 'memories'}`;
    
    
    if (filteredMemories.length === 0) {
        elements.memoryTableContainer.classList.add('hidden');
        elements.emptyState.classList.remove('hidden');
        elements.connectionErrorState.classList.add('hidden');
        elements.paginationContainer.classList.add('hidden');
        return;
    }
    
    
    elements.emptyState.classList.add('hidden');
    elements.connectionErrorState.classList.add('hidden');
    elements.memoryTableContainer.classList.remove('hidden');
    
    
    elements.memoryTableBody.innerHTML = '';
    
    
    pageMemories.forEach(memory => {
        const row = createMemoryRow(memory);
        elements.memoryTableBody.appendChild(row);
    });
    
    
    setupActionButtonListeners();
    
    updateSortIndicators();
}


function setupActionButtonListeners() {
    
    elements.memoryTableBody.removeEventListener('click', handleActionButtonClick);
    
    
    elements.memoryTableBody.addEventListener('click', handleActionButtonClick);
}


function handleActionButtonClick(event) {
    
    event.preventDefault();
    event.stopPropagation();
    
    const button = event.target.closest('.action-btn');
    if (!button) return;
    
    const action = button.dataset.action;
    const key = button.dataset.key;
    
    if (!action || !key) return;
    
    console.log('Action button clicked:', action, 'for key:', key);
    
    try {
        switch (action) {
            case 'view':
                viewMemory(key);
                break;
            case 'edit':
                editMemory(key);
                break;
            case 'delete':
                deleteMemory(key);
                break;
            default:
                console.warn('Unknown action:', action);
        }
    } catch (error) {
        console.error('Error handling action button click:', error);
        showToast('Error: ' + error.message, 'error');
    }
    
    return false;
}


function createMemoryRow(memory) {
    const row = document.createElement('tr');
    
    row.innerHTML = `
        <td class="text-truncate" title="${escapeHtml(memory.key)}">${escapeHtml(memory.key)}</td>
        <td class="text-truncate" title="${escapeHtml(memory.category)}">${escapeHtml(memory.category)}</td>
        <td>
            <span class="memory-type-badge ${memory.memory_type}">${memory.memory_type.replace('_', '-')}</span>
        </td>
        <td title="${formatFullDateTime(memory.created_at)}">${formatRelativeTime(memory.created_at)}</td>
        <td title="${formatFullDateTime(memory.updated_at)}">${formatRelativeTime(memory.updated_at)}</td>
        <td>${memory.access_count || 0}</td>
        <td class="actions-column">
            <div class="action-buttons">
                <button class="action-btn view" data-action="view" data-key="${escapeHtml(memory.key)}" title="View">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="action-btn edit" data-action="edit" data-key="${escapeHtml(memory.key)}" title="Edit">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="action-btn delete" data-action="delete" data-key="${escapeHtml(memory.key)}" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </td>
    `;
    
    return row;
}


function showLoading() {
    elements.loadingState.classList.remove('hidden');
    elements.memoryTableContainer.classList.add('hidden');
    elements.emptyState.classList.add('hidden');
    elements.connectionErrorState.classList.add('hidden');
}


function hideLoading() {
    elements.loadingState.classList.add('hidden');
}


async function refreshData() {
    
    if (isAnyModalOpen || isViewingMemory) {
        console.log('Skipping data refresh - modal state flags active (isAnyModalOpen =', isAnyModalOpen, ', isViewingMemory =', isViewingMemory, ')');
        return;
    }
    
    
    const timeSinceLastView = Date.now() - lastViewMemoryTime;
    if (timeSinceLastView < 15000) {
        console.log('Skipping data refresh - recently viewed memory (', timeSinceLastView, 'ms ago)');
        return;
    }
    
    
    const modalsVisible = [
        !elements.memoryModal.classList.contains('hidden'),
        !elements.viewModal.classList.contains('hidden'), 
        !elements.deleteModal.classList.contains('hidden'),
        
        document.querySelector('.modal:not(.hidden)') !== null
    ];
    
    if (modalsVisible.some(visible => visible)) {
        console.log('Skipping data refresh - modal is visible in DOM');
        return;
    }
    
    elements.refreshBtn.classList.add('loading');
    try {
        await Promise.all([loadMemories(), loadMemoryStatsWithRetry()]);
    } finally {
        elements.refreshBtn.classList.remove('loading');
    }
}


function openAddMemoryModal() {
    isEditMode = false;
    editingKey = null;
    isAnyModalOpen = true;
    
    elements.modalTitle.innerHTML = '<i class="fas fa-plus"></i> Add New Memory';
    elements.saveMemoryBtn.innerHTML = '<i class="fas fa-save"></i> Save Memory';
    
    
    elements.memoryForm.reset();
    elements.memoryCategory.value = 'general';
    elements.memoryType.value = 'long_term';
    elements.memoryKey.disabled = false;
    
    elements.memoryModal.classList.remove('hidden');
    elements.memoryKey.focus();
}


async function editMemory(key) {
    try {
        
        console.log('editMemory: Starting edit operation for key:', key);
        lastViewMemoryTime = Date.now();
        isViewingMemory = true;
        isAnyModalOpen = true;
        console.log('editMemory: Set modal protection flags - isViewingMemory = true, isAnyModalOpen = true');
        
        showLoadingOverlay('Loading memory...');
        
        const response = await fetch(`${API_BASE}/api/memory/${encodeURIComponent(key)}`);
        const data = await response.json();
        
        if (data.success) {
            isEditMode = true;
            editingKey = key;
            
            
            elements.modalTitle.innerHTML = '<i class="fas fa-edit"></i> Edit Memory';
            elements.saveMemoryBtn.innerHTML = '<i class="fas fa-save"></i> Update Memory';
            
            
            const memory = data.memory;
            elements.memoryKey.value = memory.key;
            elements.memoryContent.value = memory.content;
            elements.memoryCategory.value = memory.category;
            elements.memoryType.value = memory.memory_type;
            elements.memoryTags.value = Array.isArray(memory.tags) ? memory.tags.join(', ') : '';
            elements.memoryKey.disabled = true;
            
            elements.memoryModal.classList.remove('hidden');
            elements.memoryContent.focus();
            
            isViewingMemory = false;
            console.log('editMemory: Edit modal opened successfully - isViewingMemory = false, isAnyModalOpen remains true');
        } else {
            
            console.log('editMemory: API call failed, resetting modal states');
            isAnyModalOpen = false;
            isViewingMemory = false;
            throw new Error(data.message || 'Failed to load memory');
        }
    } catch (error) {
        console.error('Failed to load memory for editing:', error);
        showToast('Failed to load memory: ' + error.message, 'error');
        
        console.log('editMemory: Exception caught, resetting modal states');
        isAnyModalOpen = false;
        isViewingMemory = false;
    } finally {
        hideLoadingOverlay();
    }
}


async function viewMemory(key) {
    try {
        
        console.log('viewMemory: Starting view operation for key:', key);
        lastViewMemoryTime = Date.now();
        isViewingMemory = true;
        isAnyModalOpen = true;
        console.log('viewMemory: Set modal protection flags - isViewingMemory = true, isAnyModalOpen = true');
        showLoadingOverlay('Loading memory...');
        
        const response = await fetch(`${API_BASE}/api/memory/${encodeURIComponent(key)}`);
        const data = await response.json();
        
        if (data.success) {
            const memory = data.memory;
            
            elements.viewMemoryKey.textContent = memory.key;
            elements.viewMemoryContent.textContent = memory.content;
            elements.viewMemoryCategory.textContent = memory.category;
            elements.viewMemoryType.textContent = memory.memory_type.replace('_', '-');
            elements.viewMemoryType.className = `memory-type-badge ${memory.memory_type}`;
            elements.viewMemoryCreated.textContent = formatFullDateTime(memory.created_at);
            elements.viewMemoryUpdated.textContent = formatFullDateTime(memory.updated_at);
            elements.viewMemoryAccessCount.textContent = memory.access_count || 0;
            
            
            elements.viewMemoryTags.innerHTML = '';
            if (Array.isArray(memory.tags) && memory.tags.length > 0) {
                memory.tags.forEach(tag => {
                    const tagElement = document.createElement('span');
                    tagElement.className = 'tag-item';
                    tagElement.textContent = tag;
                    elements.viewMemoryTags.appendChild(tagElement);
                });
            } else {
                elements.viewMemoryTags.textContent = 'No tags';
            }
            
            
            elements.editFromViewBtn.onclick = () => {
                closeViewModal();
                editMemory(key);
            };
            
            elements.viewModal.classList.remove('hidden');
            
            console.log('viewMemory: View modal opened successfully - keeping modal protection flags active');
        } else {
            
            console.log('viewMemory: API call failed, resetting modal states');
            isAnyModalOpen = false;
            isViewingMemory = false;
            throw new Error(data.message || 'Failed to load memory');
        }
    } catch (error) {
        console.error('Failed to view memory:', error);
        showToast('Failed to load memory: ' + error.message, 'error');
        
        console.log('viewMemory: Exception caught, resetting modal states');
        isAnyModalOpen = false;
        isViewingMemory = false;
    } finally {
        hideLoadingOverlay();
    }
}


function deleteMemory(key) {
    console.log('deleteMemory: Called with key:', key, 'Type:', typeof key);
    
    const memory = currentMemories.find(m => m.key === key);
    if (!memory) {
        console.warn('deleteMemory: Memory not found with key:', key);
        return;
    }
    
    console.log('deleteMemory: Found memory:', memory.key);
    
    elements.deleteMemoryKey.textContent = memory.key;
    elements.deleteMemoryCategory.textContent = memory.category;
    elements.deleteMemoryType.textContent = memory.memory_type.replace('_', '-');
    
    
    elements.confirmDeleteBtn.dataset.memoryKey = key;
    console.log('deleteMemory: Stored key in button dataset:', elements.confirmDeleteBtn.dataset.memoryKey);
    
    isAnyModalOpen = true;
    elements.deleteModal.classList.remove('hidden');
}


async function confirmDelete(event) {
    
    let key;
    if (event && event.currentTarget && event.currentTarget.dataset) {
        key = event.currentTarget.dataset.memoryKey;
    } else if (typeof event === 'string') {
        
        key = event;
    }
    
    if (!key) {
        console.error('confirmDelete: No memory key provided', event);
        showToast('Error: No memory key specified', 'error');
        return;
    }
    
    console.log('confirmDelete: Deleting memory with key:', key);
    
    try {
        elements.confirmDeleteBtn.classList.add('loading');
        
        const deleteUrl = `${API_BASE}/api/memory/${encodeURIComponent(key)}`;
        console.log('confirmDelete: DELETE request to:', deleteUrl);
        
        const response = await fetch(deleteUrl, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Memory deleted successfully', 'success');
            closeDeleteModal();
            
            setTimeout(() => refreshData(), 100);
        } else {
            throw new Error(data.message || 'Failed to delete memory');
        }
    } catch (error) {
        console.error('Failed to delete memory:', error);
        showToast('Failed to delete memory: ' + error.message, 'error');
    } finally {
        elements.confirmDeleteBtn.classList.remove('loading');
    }
}


async function handleFormSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(elements.memoryForm);
    const memoryData = {
        key: formData.get('key').trim(),
        content: formData.get('content').trim(),
        category: formData.get('category').trim() || 'general',
        memory_type: formData.get('memory_type'),
        tags: formData.get('tags') ? formData.get('tags').split(',').map(t => t.trim()).filter(t => t) : null
    };
    
    
    if (!memoryData.key) {
        showToast('Memory key is required', 'error');
        return;
    }
    
    if (!memoryData.content) {
        showToast('Memory content is required', 'error');
        return;
    }
    
    try {
        elements.saveMemoryBtn.classList.add('loading');
        
        let response;
        if (isEditMode) {
            
            response = await fetch(`${API_BASE}/api/memory/${encodeURIComponent(editingKey)}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(memoryData)
            });
        } else {
            
            response = await fetch(`${API_BASE}/api/memory`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(memoryData)
            });
        }
        
        const data = await response.json();
        
        if (data.success) {
            showToast(isEditMode ? 'Memory updated successfully' : 'Memory created successfully', 'success');
            closeMemoryModal();
            
            setTimeout(() => refreshData(), 100);
        } else {
            throw new Error(data.message || 'Failed to save memory');
        }
    } catch (error) {
        console.error('Failed to save memory:', error);
        showToast('Failed to save memory: ' + error.message, 'error');
    } finally {
        elements.saveMemoryBtn.classList.remove('loading');
    }
}


function editFromView() {
    const key = elements.viewMemoryKey.textContent;
    closeViewModal();
    editMemory(key);
}


function closeMemoryModal() {
    console.log('closeMemoryModal: Closing memory modal and resetting states');
    elements.memoryModal.classList.add('hidden');
    elements.memoryForm.reset();
    isEditMode = false;
    editingKey = null;
    isAnyModalOpen = false;
    isViewingMemory = false;
    
    lastViewMemoryTime = Date.now() - 12000;
    console.log('closeMemoryModal: Modal closed, refreshes will be allowed in 3 seconds');
}


function closeDeleteModal() {
    console.log('closeDeleteModal: Closing delete modal and resetting states');
    elements.deleteModal.classList.add('hidden');
    isAnyModalOpen = false;
    isViewingMemory = false;
    
    lastViewMemoryTime = Date.now() - 12000;
    console.log('closeDeleteModal: Modal closed, refreshes will be allowed in 3 seconds');
}


function closeViewModal() {
    console.log('closeViewModal: Closing view modal and resetting states');
    elements.viewModal.classList.add('hidden');
    isAnyModalOpen = false;
    isViewingMemory = false;
    
    lastViewMemoryTime = Date.now() - 12000;
    console.log('closeViewModal: Modal closed, refreshes will be allowed in 3 seconds');
}


function closeAllModals() {
    console.log('closeAllModals: Closing all modals and resetting states');
    closeMemoryModal();
    closeDeleteModal();
    closeViewModal();
    
    isAnyModalOpen = false;
    isViewingMemory = false;
    
    lastViewMemoryTime = Date.now() - 12000;
    console.log('closeAllModals: All modals closed, refreshes will be allowed in 3 seconds');
}


function showLoadingOverlay(text = 'Processing...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.remove('hidden');
}


function hideLoadingOverlay() {
    elements.loadingOverlay.classList.add('hidden');
}


function handleKeyboardShortcuts(e) {
    
    if (e.key === 'Escape') {
        closeAllModals();
        return;
    }
    
    
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        openAddMemoryModal();
        return;
    }
    
    
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        
        
        if (isAnyModalOpen || isViewingMemory || 
            !elements.memoryModal.classList.contains('hidden') || 
            !elements.viewModal.classList.contains('hidden') || 
            !elements.deleteModal.classList.contains('hidden')) {
            console.log('Keyboard refresh blocked - modal is open');
            showToast('Cannot refresh while modal is open', 'warning', 3000);
            return;
        }
        
        refreshData();
        return;
    }
}


function showToast(message, type = 'info', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'error' ? 'fa-exclamation-circle' : 
                 type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';
    
    toast.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
        <button class="toast-close">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    
    toast.querySelector('.toast-close').addEventListener('click', () => {
        removeToast(toast);
    });
    
    elements.toastContainer.appendChild(toast);
    
    
    setTimeout(() => {
        removeToast(toast);
    }, duration);
}


function removeToast(toast) {
    if (toast && toast.parentNode) {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }
}




async function handleApiResponse(response) {
    
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Server returned HTML instead of JSON. Make sure the API is running and endpoints are available.');
    }
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
}


function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}


function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


function formatRelativeTime(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 30) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
}


function formatFullDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}


window.openAddMemoryModal = openAddMemoryModal;
window.editMemory = editMemory;
window.viewMemory = viewMemory;
window.deleteMemory = deleteMemory;
window.closeMemoryModal = closeMemoryModal;
window.closeDeleteModal = closeDeleteModal;
window.closeViewModal = closeViewModal;
window.testConnection = testConnection;