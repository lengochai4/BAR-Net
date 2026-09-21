document.addEventListener('DOMContentLoaded', () => {
    // State for 3D Viewer
    let isVolumeMode = false;
    let volumeSlices = { original: [], mask: [] };
    let currentSliceIndex = 0;
    const imageInput = document.getElementById('imageInput');
    const fileNameDisplay = document.getElementById('file-name');
    const predictBtn = document.getElementById('predictBtn');
    
    // Original Viewer
    const originalContainer = document.getElementById('originalContainer');
    const originalImg = document.getElementById('originalImg');
    
    // Result Viewer
    const resultContainer = document.getElementById('resultContainer');
    const maskTools = document.getElementById('maskTools');
    const blendWrapper = document.getElementById('blendWrapper');
    const blendBase = document.getElementById('blendBase');
    const blendMask = document.getElementById('blendMask');
    
    // Controls
    const overlayToggle = document.getElementById('overlayToggle');
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityControlContainer = document.getElementById('opacityControlContainer');
    
    const errorBox = document.getElementById('errorBox');
    
    let selectedFile = null;

    imageInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            selectedFile = file;
            fileNameDisplay.textContent = file.name;
            predictBtn.disabled = false;
            
            // Check if it's an H5 volume
            if (file.name.endsWith('.h5')) {
                isVolumeMode = true;
                originalContainer.classList.add('has-content');
                originalImg.classList.add('hidden');
                originalContainer.querySelector('.empty-content').innerHTML = '<span style="color:var(--primary); font-weight:600">3D Volume (.h5) Ready</span>';
                blendBase.src = ""; // Clear base
                
            } else {
                isVolumeMode = false;
                // Show preview of original image for 2D
                const reader = new FileReader();
                reader.onload = (e) => {
                    originalImg.src = e.target.result;
                    originalImg.classList.remove('hidden');
                    originalContainer.classList.add('has-content');
                    // Initialize blend base
                    blendBase.src = e.target.result;
                };
                reader.readAsDataURL(file);
                
                // Show standard placeholder
                originalContainer.querySelector('.empty-content').innerHTML = `
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                    <span>Please load a scan image</span>
                `;
            }
            
            // Reset result view
            blendWrapper.classList.add('hidden');
            maskTools.classList.add('hidden');
            resultContainer.classList.remove('has-content');
            errorBox.classList.add('hidden');
            overlayToggle.checked = true; // Default to overlay on next run
            
            // Remove any slice slider if present
            const oldSlider = document.getElementById('sliceScrollerContainer');
            if (oldSlider) oldSlider.remove();

        } else {
            selectedFile = null;
            fileNameDisplay.textContent = "No file selected";
            predictBtn.disabled = true;
        }
    });

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI updates for loading state
        const btnText = predictBtn.querySelector('.btn-text');
        const loader = predictBtn.querySelector('.loader');
        
        predictBtn.disabled = true;
        btnText.textContent = "Processing Scan...";
        loader.classList.remove('hidden');
        errorBox.classList.add('hidden');
        
        maskTools.classList.add('hidden');
        blendWrapper.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const endpoint = isVolumeMode ? '/predict_volume' : '/predict';
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                if (isVolumeMode) {
                    volumeSlices.original = data.original_slices;
                    volumeSlices.mask = data.mask_slices;
                    currentSliceIndex = Math.floor(data.num_slices / 2); // Start at middle
                    
                    // Setup UI for volume
                    setupVolumeControls(data.num_slices);
                    updateVolumeView();
                    
                    // Show standard placeholder text just in case (hidden by has-content)
                    originalContainer.querySelector('.empty-content').innerHTML = `
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                        <span>Please load a scan image</span>
                    `;
                } else {
                    const maskSrc = `data:image/png;base64,${data.mask}`;
                    blendMask.src = maskSrc;
                }
                
                // Show tools and results
                maskTools.classList.remove('hidden');
                resultContainer.classList.add('has-content');
                blendWrapper.classList.remove('hidden');
                
                // Apply current opacity
                blendMask.style.opacity = opacitySlider.value;
                if (overlayToggle.checked) {
                    blendBase.classList.remove('hidden');
                } else {
                    blendBase.classList.add('hidden');
                }

            } else {
                throw new Error(data.message || "An error occurred during prediction.");
            }
            
        } catch (error) {
            console.error('Prediction Error:', error);
            errorBox.textContent = error.message;
            errorBox.classList.remove('hidden');
        } finally {
            predictBtn.disabled = false;
            btnText.textContent = "Run BAR-Net Segmentation";
            loader.classList.add('hidden');
        }
    });

    // Overlay logic
    overlayToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            blendBase.classList.remove('hidden');
            opacityControlContainer.classList.remove('hidden');
        } else {
            blendBase.classList.add('hidden'); // Hide base to show only mask
            opacityControlContainer.classList.add('hidden');
            blendMask.style.opacity = 1; // Full visibility for mask-only view
        }
    });

    opacitySlider.addEventListener('input', (e) => {
        if (overlayToggle.checked) {
            blendMask.style.opacity = e.target.value;
        }
    });

    // Helper to setup volume controls dynamically
    function setupVolumeControls(totalSlices) {
        // Remove old if exists
        const oldScroller = document.getElementById('sliceScrollerContainer');
        if (oldScroller) oldScroller.remove();

        const scrollerHTML = `
            <div id="sliceScrollerContainer" class="volume-slider-container">
                <span style="font-size:0.8rem; color:var(--text-secondary);">Slice Viewer</span>
                <input type="range" id="sliceSlider" min="0" max="${totalSlices - 1}" step="1" value="${currentSliceIndex}" style="flex:1;">
                <span id="sliceLabel" style="font-size:0.8rem; width:50px; text-align:right;">${currentSliceIndex+1}/${totalSlices}</span>
            </div>
        `;
        
        // Append below the errorBox or above viewer grid
        const workspaceHeader = document.querySelector('.workspace-header');
        workspaceHeader.insertAdjacentHTML('afterend', scrollerHTML);
        
        document.getElementById('sliceSlider').addEventListener('input', (e) => {
            currentSliceIndex = parseInt(e.target.value);
            document.getElementById('sliceLabel').textContent = `${currentSliceIndex+1}/${totalSlices}`;
            updateVolumeView();
        });
    }

    // Update image srcs based on slice index
    function updateVolumeView() {
        if (!isVolumeMode) return;
        const origData = `data:image/png;base64,${volumeSlices.original[currentSliceIndex]}`;
        const maskData = `data:image/png;base64,${volumeSlices.mask[currentSliceIndex]}`;
        
        // Update both the original preview block and the blendBase
        originalImg.src = origData;
        originalImg.classList.remove('hidden');
        
        blendBase.src = origData;
        blendMask.src = maskData;
    }
});
