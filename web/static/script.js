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

    // Set current date automatically
    const dateInput = document.getElementById('dateInput');
    if (dateInput) {
        const today = new Date();
        const yyyy = today.getFullYear();
        const mm = String(today.getMonth() + 1).padStart(2, '0');
        const dd = String(today.getDate()).padStart(2, '0');
        dateInput.value = `${yyyy}-${mm}-${dd}`;
    }

    // Sync Doctor info to header visually
    const docNameInp = document.getElementById('doctorNameInput');
    const docRoleInp = document.getElementById('doctorRoleInput');
    const headerName = document.querySelector('.user-details .name');
    const headerRole = document.querySelector('.user-details .role');
    
    if (docNameInp && headerName) {
        docNameInp.addEventListener('input', e => headerName.textContent = e.target.value);
    }
    if (docRoleInp && headerRole) {
        docRoleInp.addEventListener('input', e => headerRole.textContent = e.target.value);
    }

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
            
            // AUTOMATICALLY RUN PREDICTION ON UPLOAD
            runPrediction();

        } else {
            selectedFile = null;
            fileNameDisplay.textContent = "No file selected";
            predictBtn.disabled = true;
        }
    });

    predictBtn.addEventListener('click', () => {
        runPrediction();
    });
    
    async function runPrediction() {
        if (!selectedFile) return;

        // UI updates for loading state
        const btnText = predictBtn.querySelector('.btn-text');
        const loader = predictBtn.querySelector('.loader');
        
        predictBtn.disabled = true;
        
        if (btnText) btnText.textContent = "Processing Scan...";
        else predictBtn.innerText = "Processing Scan...";
        
        if (loader) loader.classList.remove('hidden');
        
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
                    // Baked image: show directly in blendBase, no overlay needed
                    blendBase.src = maskSrc;
                    blendMask.src = '';
                    blendMask.style.opacity = 0;
                }
                
                // Show results
                maskTools.classList.remove('hidden');
                resultContainer.classList.add('has-content');
                blendWrapper.classList.remove('hidden');

            } else {
                throw new Error(data.message || "An error occurred during prediction.");
            }
            
        } catch (error) {
            console.error('Prediction Error:', error);
            errorBox.textContent = error.message;
            errorBox.classList.remove('hidden');
            predictBtn.disabled = false; // re-enable on error
        } finally {
            if (btnText) btnText.textContent = isVolumeMode ? "Loaded 3D Volume" : "Loaded Image";
            else predictBtn.innerText = isVolumeMode ? "Loaded 3D Volume" : "Loaded Image";
            
            if (loader) loader.classList.add('hidden');
        }
    }

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
        
        // LEFT panel: pure original CT
        originalImg.src = origData;
        originalImg.classList.remove('hidden');
        originalContainer.classList.add('has-content');
        
        // RIGHT panel: blend using canvas
        blendToCanvas(origData, maskData);
    }

    function blendToCanvas(origSrc, maskSrc) {
        // Get or create canvas inside blendWrapper
        let canvas = document.getElementById('blendCanvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.id = 'blendCanvas';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            canvas.style.objectFit = 'contain';
            blendWrapper.innerHTML = ''; // clear old img tags
            blendWrapper.appendChild(canvas);
        }
        
        const origImg = new Image();
        origImg.onload = function() {
            canvas.width = origImg.naturalWidth;
            canvas.height = origImg.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Draw original CT
            ctx.drawImage(origImg, 0, 0);
            // Draw mask on top with 70% opacity
            const maskImg = new Image();
            maskImg.onload = function() {
                ctx.globalAlpha = 0.70;
                ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
                ctx.globalAlpha = 1.0;
            };
            maskImg.src = maskSrc;
        };
        origImg.src = origSrc;
    }

    // Fullscreen / Zoom logic
    const zoomBtns = document.querySelectorAll('.zoom-btn');
    zoomBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const panel = e.target.closest('.panel');
            if (panel) {
                panel.classList.toggle('fullscreen');
                // Change icon
                const isFs = panel.classList.contains('fullscreen');
                if (isFs) {
                    btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path></svg>';
                } else {
                    btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>';
                }
            }
        });
    });

    // --------------------------------------------------------
    // Export Medical Report to PNG
    // --------------------------------------------------------
    const saveReportBtn = document.getElementById('saveReportBtn');
    if(saveReportBtn) {
        saveReportBtn.addEventListener('click', generateReport);
    }
    
    function generateReport() {
        if (!originalImg.src) {
            alert('Please process an image first before exporting.');
            return;
        }

        const patientId = document.getElementById('patientIdInput').value || 'UnknownID';
        const patientName = document.getElementById('patientNameInput').value || 'UnknownName';
        const age = document.getElementById('patientAgeInput').value || '--';
        const gender = document.getElementById('patientGenderInput').value || '--';
        const modalities = document.getElementById('modalityInput').value || '--';
        const date = document.getElementById('dateInput').value || '--';
        const obs = document.getElementById('obsTextArea').value || 'No clinical observations provided.';
        
        const docId = document.getElementById('doctorIdInput').value || 'MEC-000';
        const docName = document.getElementById('doctorNameInput').value || 'Unknown';
        const docRole = document.getElementById('doctorRoleInput').value || 'Doctor';

        // Layout Dimensions (UHQ Landscape)
        const width = 2400;
        const height = 1800;
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        // Draw background
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        // Header Title
        ctx.fillStyle = '#3b82f6';
        ctx.font = 'bold 50px Outfit, Arial';
        ctx.textAlign = 'center';
        ctx.fillText('MEDICAL IMAGING REPORT - MedVision AI', width/2, 90);

        // Info Section
        ctx.fillStyle = '#f8fafc';
        ctx.font = '30px Outfit, Arial';
        ctx.textAlign = 'left';
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString('vi-VN');

        let y = 180;
        ctx.fillText(`Patient ID: ${patientId}`, 100, y);
        ctx.fillText(`Date: ${date} (Time: ${timeStr})`, 1250, y); y += 50;
        ctx.fillText(`Patient Name: ${patientName}`, 100, y);
        ctx.fillText(`Modality: ${modalities}`, 1250, y); y += 50;
        ctx.fillText(`Gender: ${gender}        Age: ${age}`, 100, y);
        ctx.fillText(`Physician: ${docName} - ${docRole} (${docId})`, 1250, y);
        
        ctx.beginPath();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 3;
        ctx.moveTo(100, y + 40);
        ctx.lineTo(2300, y + 40);
        ctx.stroke();

        y += 110;
        // Images Section
        ctx.font = 'bold 36px Outfit, Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Original Scan`, 625, y);
        ctx.fillText(`AI Segmentation (BAR-Net)`, 1775, y);
        
        y += 40;

        // Draw Images Configuration (Full Frame horizontally)
        const imgSize = 1050; // Nearly fills the entire 2400px width with two images
        const margin1 = 100;
        const margin2 = 1250;
        
        // Draw Original
        if (originalImg.src) {
            ctx.drawImage(originalImg, margin1, y, imgSize, imgSize);
        }
        
        // Draw Segmentation
        ctx.fillStyle = '#0a1120';
        ctx.fillRect(margin2, y, imgSize, imgSize);
        
        // Fetch segmentation image depending on mode
        const segmentationSource = isVolumeMode ? document.getElementById('blendCanvas') : blendBase;
        if (segmentationSource && (segmentationSource.src || isVolumeMode)) {
            ctx.drawImage(segmentationSource, margin2, y, imgSize, imgSize);
        }

        y += imgSize + 90;
        
        // Observations
        ctx.textAlign = 'left';
        ctx.fillStyle = '#3b82f6';
        ctx.font = 'bold 32px Outfit, Arial';
        ctx.fillText('Clinical Observations', 100, y);
        y += 50;
        
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '28px Outfit, Arial';
        
        // Multi-line word wrap
        const words = obs.split(' ');
        let line = '';
        const maxWidth = 2200;
        
        for(let n = 0; n < words.length; n++) {
            const testLine = line + words[n] + ' ';
            const metrics = ctx.measureText(testLine);
            if (metrics.width > maxWidth && n > 0) {
                ctx.fillText(line, 100, y);
                line = words[n] + ' ';
                y += 40;
            } else {
                line = testLine;
            }
        }
        ctx.fillText(line, 100, y);

        // Force download
        let safeId = patientId.replace(/[\\/:*?"<>|]/g, '_').trim();
        let safeName = patientName.replace(/[\\/:*?"<>|]/g, '_').trim();
        if (!safeId) safeId = 'Scan';
        if (!safeName) safeName = 'Report';
        
        const dataUrl = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `${safeId}_${safeName}.png`;
        a.click();
    }
});
