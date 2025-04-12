document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const detectionMode = document.getElementById('detection-mode');
    const segmentationOptions = document.getElementById('segmentation-options');
    const uploadForm = document.getElementById('uploadForm');

    // Update file input label when a file is selected
    fileInput.addEventListener('change', (e) => {
        const fileName = e.target.files[0] ? e.target.files[0].name : 'Choose an image';
        const uploadText = document.querySelector('.upload-text');
        uploadText.textContent = fileName;
    });

    // Toggle segmentation model options based on detection mode
    detectionMode.addEventListener('change', (e) => {
        if (e.target.value === 'segmentation') {
            segmentationOptions.classList.remove('hidden');
        } else {
            segmentationOptions.classList.add('hidden');
        }
    });

    // Optional: Client-side form validation
    uploadForm.addEventListener('submit', (e) => {
        const fileInput = document.getElementById('fileInput');
        
        if (!fileInput.files.length) {
            e.preventDefault();
            alert('Please select an image to upload.');
        }
    });
});