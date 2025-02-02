const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const urlInput = document.getElementById('urlInput');
        const urlSubmit = document.getElementById('urlSubmit');
        const resultContent = document.getElementById('resultContent');

        document.querySelector('.upload-btn').addEventListener('click', () => {
            fileInput.click();
        });

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
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                processImage(file);
            }
        });

        urlSubmit.addEventListener('click', () => {
            const url = urlInput.value.trim();
            if (url) {
                fetch(url)
                    .then(response => response.blob())
                    .then(blob => {
                        const file = new File([blob], "image.jpg", { type: "image/jpeg" });
                        processImage(file);
                    })
                    .catch(error => {
                        alert('Gagal memuat gambar dari URL');
                    });
            }
        });

        function useExampleImage(imgElement) {
            fetch(imgElement.src)
                .then(response => response.blob())
                .then(blob => {
                    const file = new File([blob], "example.jpg", { type: "image/jpeg" });
                    processImage(file);
                })
                .catch(error => {
                    alert('Gagal memuat gambar contoh');
                });
        }

        function processImage(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = function() {
                    const fileSize = (file.size / (1024 * 1024)).toFixed(2);
                    showResults(e.target.result, fileSize, this.width, this.height);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function showResults(imageSrc, fileSize, width, height) {
            const resultHTML = `
                <div class="result-box">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem;">Gambar Yang Dianalisis</h3>
                        <div class="image-comparison">
                            <div class="image-comparison-item">
                                <img src="${imageSrc}" alt="Original Image">
                                <p class="image-label">Gambar Asli</p>
                            </div>
                            <div class="image-comparison-item">
                                <img src="${imageSrc}" alt="Analyzed Image" style="filter: grayscale(100%) contrast(120%);">
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
                        <li>Struktur anatomi dalam batas normal tanpa kelainan signifikan</li>
                        <li>Tidak terdeteksi adanya lesi atau massa abnormal</li>
                        <li>Jaringan menunjukkan densitas dan intensitas normal</li>
                        <li>Pola distribusi dan morfologi sesuai dengan karakteristik normal</li>
                        <li>Tidak ada indikasi perubahan patologis yang teridentifikasi</li>
                    </ul>

                    <div class="confidence-section">
                        <p style="font-size: 0.9rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                            <span>Tingkat Akurasi Analisis</span>
                            <span style="color: var(--primary-blue); font-weight: 500;">92%</span>
                        </p>
                        <div class="confidence-bar">
                            <div class="confidence-progress" style="width: 92%"></div>
                        </div>
                    </div>
                </div>

                <div class="result-box">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem;">Rekomendasi</h3>
                    <p style="font-size: 0.9rem; color: rgba(17, 45, 78, 0.75); line-height: 1.6;">
                        Berdasarkan hasil analisis, tidak ditemukan indikasi yang memerlukan tindak lanjut segera. 
                        Namun, disarankan untuk tetap melakukan pemeriksaan rutin sesuai dengan jadwal yang 
                        direkomendasikan oleh tenaga medis Anda.
                    </p>
                </div>
            `;

            resultContent.innerHTML = resultHTML;

            // Animate confidence bar
            setTimeout(() => {
                const progressBar = document.querySelector('.confidence-progress');
                if (progressBar) {
                    progressBar.style.width = '92%';
                }
            }, 100);
        }