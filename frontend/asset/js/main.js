    // Improved JavaScript code for handling image uploads
    document.addEventListener('DOMContentLoaded', function() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const urlInput = document.getElementById('urlInput');
        const urlSubmit = document.getElementById('urlSubmit');
        const resultContent = document.getElementById('resultContent');
        const selectFileBtn = document.getElementById('selectFileBtn');
        const uploadForm = document.getElementById('uploadForm');
    
        // Variable to store the original image URL
        let originalImageUrl = '';
    
        // Handle file selection button click
        selectFileBtn.addEventListener('click', () => {
            fileInput.click();
        });
    
        // Handle drag and drop events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
    
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                processImage(file);
            } else {
                alert('Please upload an image file');
            }
        });
    
        // Handle file input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                processImage(file);
            } else {
                alert('Please select an image file');
            }
        });
    
        // Handle URL submission
        urlSubmit.addEventListener('click', () => {
            const url = urlInput.value.trim();
            if (url) {
                fetch(url)
                    .then(response => {
                        if (!response.ok) throw new Error('Network response was not ok');
                        return response.blob();
                    })
                    .then(blob => {
                        const file = new File([blob], "image.jpg", { type: "image/jpeg" });
                        originalImageUrl = URL.createObjectURL(blob);
                        processImage(file);
                    })
                    .catch(error => {
                        console.error('Error fetching image from URL:', error);
                        alert('Failed to load image from URL');
                    });
            }
        });
    
        // Handle example image clicks
        document.querySelectorAll('.example-image').forEach(img => {
            img.addEventListener('click', function() {
                originalImageUrl = this.src;
                fetch(this.src)
                    .then(response => {
                        if (!response.ok) throw new Error('Network response was not ok');
                        return response.blob();
                    })
                    .then(blob => {
                        const file = new File([blob], "example.jpg", { type: "image/jpeg" });
                        processImage(file);
                    })
                    .catch(error => {
                        console.error('Error loading example image:', error);
                        alert('Failed to load example image');
                    });
            });
        });
    
        function processImage(file) {
            // Show loading state with spinner
            resultContent.innerHTML = `
                <div class="result-box">
                    <div style="text-align: center; padding: 20px;">
                        <div style="display: inline-block; width: 40px; height: 40px; border: 3px solid #f3f3f3; 
                             border-top: 3px solid #3F72AF; border-radius: 50%; animation: spin 1s linear infinite;">
                        </div>
                        <p style="margin-top: 10px;">Processing image...</p>
                    </div>
                </div>
            `;
    
            // Create a preview of the original image
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = function() {
                    if (!originalImageUrl) {
                        originalImageUrl = e.target.result;
                    }
                    console.log('Image loaded successfully, proceeding with backend processing');
                    sendImageToBackend(file);
                };
                img.onerror = function() {
                    console.error('Error loading image preview');
                    alert('Error loading image');
                };
                img.src = e.target.result;
            };
            reader.onerror = function(error) {
                console.error('Error reading file:', error);
                alert('Error reading file');
            };
            reader.readAsDataURL(file);
        }
    
        function sendImageToBackend(file) {
            console.log('Sending image to backend...');
            const formData = new FormData();
            formData.append('file', file);
    
            // Log the form data being sent
            for (let pair of formData.entries()) {
                console.log('FormData entry:', pair[0], pair[1]);
            }
    
            fetch('http://127.0.0.1:8000/predict/', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('Backend response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Backend response data:', data);
                
                // Validate the response structure
                if (!data.result) {
                    throw new Error('Invalid response structure: missing result object');
                }
    
                // Handle relative URLs for processed image
                if (data.result.url_image_output && !data.result.url_image_output.startsWith('http')) {
                    data.result.url_image_output = 'http://127.0.0.1:8000' + data.result.url_image_output;
                }
    
                // Add the original image URL to the data
                data.result.originalImageUrl = originalImageUrl;
                
                // Store in localStorage
                localStorage.setItem('predictionResult', JSON.stringify(data));
                
                // Display the results
                displayPredictionResults(data);
                
                // Reset the originalImageUrl after displaying
                originalImageUrl = '';
                
                console.log('Processing complete, results displayed');
            })
            .catch(error => {
                console.error('Error processing image:', error);
                resultContent.innerHTML = `
                    <div class="result-box">
                        <h3 style="font-size: 1rem; color: red;">Error</h3>
                        <p>Failed to process image: ${error.message}</p>
                        <p style="font-size: 0.8rem; margin-top: 10px;">Please try again or contact support if the problem persists.</p>
                    </div>
                `;
                originalImageUrl = '';
            });
        }
    
        function displayPredictionResults(data) {
            console.log('Displaying prediction results:', data);
    
            // Validate required data
            if (!data.result.originalImageUrl || !data.result.url_image_output) {
                console.error('Missing image URLs in data:', data);
                resultContent.innerHTML = `
                    <div class="result-box">
                        <h3 style="font-size: 1rem; color: red;">Error</h3>
                        <p>Failed to load images. Missing image URLs.</p>
                        <pre style="font-size: 0.8rem; margin-top: 10px; overflow: auto;">
                            Original Image: ${data.result.originalImageUrl || 'missing'}
                            Processed Image: ${data.result.url_image_output || 'missing'}
                        </pre>
                    </div>
                `;
                return;
            }
    
            const resultHTML = `
                <div class="result-box">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem;">Gambar Yang Dianalisis</h3>
                    <div class="image-comparison">
                        <div class="image-comparison-item">
                            <img src="${data.result.originalImageUrl}" 
                                 alt="Original Image" 
                                 onload="console.log('Original image loaded successfully')"
                                 onerror="this.onerror=null; this.src='asset/img/error-image.jpg'; console.error('Failed to load original image');">
                            <p class="image-label">Gambar Asli</p>
                        </div>
                        <div class="image-comparison-item">
                            <img src="${data.result.url_image_output}" 
                                 alt="Analyzed Image" 
                                 onload="console.log('Analyzed image loaded successfully')"
                                 onerror="this.onerror=null; this.src='asset/img/error-image.jpg'; console.error('Failed to load analyzed image');">
                            <p class="image-label">Hasil Scan</p>
                        </div>
                    </div>
                </div>
    
                <div class="result-box">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem;">Hasil Analisis</h3>
                    <p style="font-size: 0.9rem; margin-bottom: 1rem; color: rgba(17, 45, 78, 0.75);">
                        Berdasarkan analisis terhadap gambar yang diunggah, berikut adalah temuan yang teridentifikasi:
                    </p>
                    <ul class="findings-list">
                        <li>Jenis Tumor: ${data.result.terdeteksi || 'Tidak terdeteksi'}</li>
                        <li>Confidence: ${((data.result.confidence || 0) * 100).toFixed(2)}%</li>
                        <li>Waktu Proses: ${data.result.processing_time || 'N/A'}</li>
                    </ul>
    
                    <div class="confidence-section">
                        <p style="font-size: 0.9rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                            <span>Tingkat Akurasi Analisis</span>
                            <span style="color: var(--primary-blue); font-weight: 500;">
                                ${((data.result.confidence || 0) * 100).toFixed(2)}%
                            </span>
                        </p>
                        <div class="confidence-bar">
                            <div class="confidence-progress" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
    
                <div class="result-box">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem;">Rekomendasi</h3>
                    <p style="font-size: 0.9rem; color: rgba(17, 45, 78, 0.75); line-height: 1.6;">
                        Berdasarkan hasil analisis, disarankan untuk berkonsultasi dengan dokter spesialis untuk tindak lanjut lebih lanjut.
                    </p>
                </div>
            `;
    
            resultContent.innerHTML = resultHTML;
    
            // Animate confidence bar
            setTimeout(() => {
                const progressBar = document.querySelector('.confidence-progress');
                if (progressBar) {
                    progressBar.style.width = `${((data.result.confidence || 0) * 100).toFixed(2)}%`;
                }
            }, 100);
        }
    
        // Add keypress event for URL input
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                urlSubmit.click();
            }
        });
    
        // Check for stored results on page load
        const storedResult = localStorage.getItem('predictionResult');
        if (storedResult) {
            try {
                const result = JSON.parse(storedResult);
                displayPredictionResults(result);
            } catch (error) {
                console.error('Error loading stored results:', error);
                localStorage.removeItem('predictionResult');
            }
        }
    
        // Add CSS for loading spinner animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    });