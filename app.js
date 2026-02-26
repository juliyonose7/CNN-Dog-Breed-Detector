// Base URL where the classification model API is served
// FastAPI serves the ResNet50 model on localhost port 8000
const API_BASE_URL = 'http://localhost:8000';

// Object that maintains the global application state
// Centralizes all critical runtime information
const state = {
    currentImage: null,     // currently selected image file
    isLoading: false,      // indicates whether an async operation is in progress
    results: null          // stores the results of the last classification request
};

// References to DOM elements for efficient manipulation
// Avoids repeated getElementById lookups during runtime
const elements = {
    uploadArea: null,        // drag-and-drop / click zone for image selection
    fileInput: null,         // hidden file input that handles file selection
    imagePreview: null,      // container that renders the image preview
    previewImg: null,        // <img> element that renders the uploaded image
    resetBtn: null,          // button to clear state and upload a new image
    loadingContainer: null,  // visual loading indicator shown during inference
    resultsContainer: null   // area where classification results are rendered
};

// Fires automatically when the DOM is fully parsed and ready
// Ensures all HTML elements are available before any manipulation
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();    // cache references to all required DOM elements
    setupEventListeners();   // attach all user interaction event handlers
    initParticles();        // start the background particle animation
    checkAPIConnection();    // verify that the model API is reachable
});

// Cache references to all important HTML elements
// Improves performance by avoiding repeated DOM traversal
function initializeElements() {
    elements.uploadArea = document.getElementById('uploadArea');           // drag-and-drop upload zone
    elements.fileInput = document.getElementById('fileInput');             // hidden input for file selection
    elements.imagePreview = document.getElementById('imagePreview');       // image preview container
    elements.previewImg = document.getElementById('previewImg');           // displayed preview image element
    elements.resetBtn = document.getElementById('resetBtn');               // reset / new upload button
    elements.loadingContainer = document.getElementById('loadingContainer'); // loading indicator element
    elements.resultsContainer = document.getElementById('resultsContainer'); // results display area
}

// Attach all event handlers that drive user interaction
// Defines how the UI responds to clicks, drag-and-drop, and file input
function setupEventListeners() {
    // Click on the upload zone triggers the hidden file selector
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    
    // Drag-and-drop events for dragging images from the file explorer
    elements.uploadArea.addEventListener('dragover', handleDragOver);   // fires while a file is dragged over the zone
    elements.uploadArea.addEventListener('dragleave', handleDragLeave); // fires when the dragged file leaves the zone
    elements.uploadArea.addEventListener('drop', handleDrop);           // fires when a file is dropped onto the zone

    // Fires when the user selects a file through the file dialog
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Resets the interface and allows uploading a new image
    elements.resetBtn.addEventListener('click', resetInterface);

    // Prevent default browser drag-and-drop behavior
    // Required for the custom drag-and-drop implementation to work correctly
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.uploadArea.addEventListener(eventName, preventDefaults);
        document.body.addEventListener(eventName, preventDefaults);
    });
}

// Prevents default browser actions for drag events
// Required to implement custom drag-and-drop behavior
function preventDefaults(e) {
    e.preventDefault();  // cancel the default browser action
    e.stopPropagation(); // prevent the event from bubbling up to parent elements
}

// Handles the visual feedback when a file is dragged over the upload zone
// Applies an active style to indicate the drop target is ready
function handleDragOver(e) {
    elements.uploadArea.classList.add('dragover'); // apply active drop zone style
}

// Handles cleanup when a dragged file leaves the upload zone
// Removes the active drop zone visual indicator
function handleDragLeave(e) {
    elements.uploadArea.classList.remove('dragover'); // remove active drop zone style
}

// Handles a file dropped onto the upload zone
// Processes the first dropped file
function handleDrop(e) {
    elements.uploadArea.classList.remove('dragover'); // removes visual style
    const files = e.dataTransfer.files;               // retrieve the dropped file list
    if (files.length > 0) {
        processFile(files[0]); // process only the first file
    }
}

// Handles file selection from the file dialog
// Processes the user-selected file from the input element
function handleFileSelect(e) {
    const file = e.target.files[0]; // retrieve the first selected file
    if (file) {
        processFile(file); // process the selected file
    }
}

// Validates and processes the selected file before sending it for classification
// Performs type and size validation before proceeding
function processFile(file) {
    // Verify that the file is a valid image type
    // MIME type check using the file.type prefix
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return; // abort if not an image
    }

    // Implementation note.
    // 10 * 1024 * 1024 = 10,485,760 bytes = 10 MB limit
    if (file.size > 10 * 1024 * 1024) {
        showError('Image exceeds the maximum allowed size of 10 MB');
        return; // abort if file exceeds size limit
    }

    state.currentImage = file;    // store a reference to the current file in global state
    displayImagePreview(file);    // render a preview of the selected image
    classifyImage(file);         // send the image to the API for breed classification
}

// Renders a preview of the selected image before inference
// Converts the file to a base64 data URL renderable by the browser
function displayImagePreview(file) {
    const reader = new FileReader(); // browser FileReader API for reading local files
    
    // Callback fired when FileReader finishes reading the file
    reader.onload = function(e) {
        elements.previewImg.src = e.target.result; // assign the data URL to the <img> element
        showSection('imagePreview');               // show the preview section
        hideSection('uploadArea');                 // hide the original upload zone
    };
    
    // Start reading the file as a base64-encoded data URL
    // Converts the image to a data URL that the browser can render directly
    reader.readAsDataURL(file);
}

// Sends the image to the model API for breed classification
// Manages the full HTTP request/response cycle with the backend
async function classifyImage(file) {
    try {
        showSection('loadingContainer');  // display the loading indicator
        hideSection('resultsContainer');  // hide any previous results
        state.isLoading = true;          // flag global state as processing

        // Build multipart/form-data payload required by the FastAPI endpoint
        const formData = new FormData();
        formData.append('file', file);   // attach the image file under the key 'file'

        console.log('🔄 Sending image to the API...'); // log for debugging
        
        // Perform HTTP POST to the prediction endpoint
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData  // omit Content-Type; fetch sets it automatically for FormData
        });

        // Check whether the HTTP response indicates success
        if (!response.ok) {
            const errorText = await response.text(); // read raw error body for diagnostics
            console.error('API error response:', errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        // Parse the JSON response body into a JavaScript object
        const results = await response.json();
        console.log('✅ API response:', results); // log for debugging
        
        // Verify that the API returned a well-formed success response
        if (!results || !results.success) {
            throw new Error('The API returned an error response');
        }

        state.results = results;  // persist results in global state
        displayResults(results);  // render results to the user

    } catch (error) {
        console.error('❌ Classification error:', error);
        showError(`Classification failed: ${error.message}`);
    } finally {
        state.isLoading = false;         // clear the processing flag
        hideSection('loadingContainer'); // hide the loading indicator
    }
}

// Builds and renders the results UI with the model predictions
// Dynamically creates HTML elements for an attractive results layout
function displayResults(results) {
    const container = elements.resultsContainer;
    
    container.innerHTML = ''; // clear any previous results

    // Validate that the API response has the expected structure
    // Guard against malformed or unexpected API responses
    if (!results || !results.top_predictions || !Array.isArray(results.top_predictions)) {
        showError('Error: Invalid API response format');
        return; // abort rendering if data is invalid
    }

    // Create the main result card container
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card'; // apply result card CSS class

    // Build the header section with the top prediction
    const header = document.createElement('div');
    header.className = 'result-header';
    
    // Element to display the most probable breed
    const title = document.createElement('h2');
    title.className = 'result-title';
    
    // Extract the top breed from the response with a safe fallback
    const mainBreed = results.recommendation?.most_likely || results.top_predictions[0]?.breed || 'Desconocido';
    title.textContent = `🐕 ${formatBreedName(mainBreed)}`;
    
    // Create a confidence badge with color-coded confidence level
    const confidenceBadge = document.createElement('div');
    
    // Extract the top confidence score with a safe fallback
    const mainConfidence = results.recommendation?.confidence || results.top_predictions[0]?.confidence || 0;
    confidenceBadge.className = `confidence-badge ${getConfidenceLevel(mainConfidence)}`;
    confidenceBadge.textContent = `${(mainConfidence * 100).toFixed(1)}% confidence`;
    
    // Assemble the header section
    header.appendChild(title);
    header.appendChild(confidenceBadge);

    // Build the ranked predictions list
    const predictionsList = document.createElement('div');
    predictionsList.className = 'predictions-list';

    // Limit display to the top 5 predictions to avoid UI clutter
    const topPredictions = results.top_predictions.slice(0, 5);
    topPredictions.forEach((prediction, index) => {
        const item = createPredictionItem(prediction, index + 1); // 1-based rank for display
        predictionsList.appendChild(item);
    });

    // Assemble the complete result card
    resultCard.appendChild(header);
    resultCard.appendChild(predictionsList);
    container.appendChild(resultCard);

    showSection('resultsContainer'); // make the results section visible
}

// Creates an individual visual element for a single breed prediction
// Includes rank badge, breed name, confidence bar, and percentage
function createPredictionItem(prediction, rank) {
    // Main item container with a special style for the top prediction
    const item = document.createElement('div');
    item.className = `prediction-item ${rank === 1 ? 'top' : ''}`; // special class for rank #1

    // Left section: rank badge and breed name
    const breedInfo = document.createElement('div');
    breedInfo.className = 'breed-info';

    // Circular badge displaying the rank number
    const rankBadge = document.createElement('div');
    rankBadge.className = 'breed-rank';
    rankBadge.textContent = rank; // display 1, 2, 3, etc.

    // Breed name formatted for human readability
    const breedName = document.createElement('div');
    breedName.className = 'breed-name';
    breedName.textContent = formatBreedName(prediction.breed);

    // Assemble the breed info section
    breedInfo.appendChild(rankBadge);
    breedInfo.appendChild(breedName);

    // Right section: confidence score visualization
    const confidenceInfo = document.createElement('div');
    confidenceInfo.className = 'confidence-info';

    // Progress bar container
    const confidenceBar = document.createElement('div');
    confidenceBar.className = 'confidence-bar';

    // Progress bar fill proportional to the confidence percentage
    const confidenceFill = document.createElement('div');
    confidenceFill.className = 'confidence-fill';
    confidenceFill.style.width = `${prediction.confidence * 100}%`; // width proportional to confidence

    confidenceBar.appendChild(confidenceFill);

    // Numeric confidence percentage label
    const confidencePercent = document.createElement('div');
    confidencePercent.className = 'confidence-percent';
    confidencePercent.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    // Assemble the confidence section
    confidenceInfo.appendChild(confidenceBar);
    confidenceInfo.appendChild(confidencePercent);

    // Assemble the complete prediction item
    item.appendChild(breedInfo);
    item.appendChild(confidenceInfo);

    return item; // return the element ready for DOM insertion
}

// Converts technical breed identifiers to human-readable display names
// Transforms 'golden_retriever' into 'Golden Retriever'
function formatBreedName(breedName) {
    return breedName
        .split('_')                                    // split on underscores
        .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // capitalize each word
        .join(' ');                                    // rejoin with spaces
}

// Determines the confidence tier for applying the appropriate color class
// Helps users quickly interpret the model certainty level
function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'high';    // green: high confidence (>=80%)
    if (confidence >= 0.5) return 'medium';  // yellow: medium confidence (50–79%)
    return 'low';                             // red: low confidence (<50%)
}

// Resets the entire UI to its initial state for a new classification
// Clears all previous data and restores the upload view
function resetInterface() {
    state.currentImage = null;  // clear the previous file reference
    state.results = null;       // clear previous results
    
    elements.fileInput.value = '';    // reset the file input value
    elements.previewImg.src = '';     // clear the displayed preview image
    
    // Control section visibility
    hideSection('imagePreview');      // hide the image preview
    hideSection('loadingContainer');  // hide the loading indicator
    hideSection('resultsContainer');  // hide previous results
    showSection('uploadArea');        // show the upload zone
}

// Makes a specific UI section visible
// Centralizes visibility control for consistency
function showSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'block'; // reveal element with display:block
    }
}

// Hides a specific UI section
// Centralizes visibility control for consistency
function hideSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.style.display = 'none'; // conceal element with display:none
    }
}

// Creates a temporary toast notification that auto-dismisses after 5 seconds
function showError(message) {
    // Create the toast notification element
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    
    // Apply inline CSS for positioning and appearance
    toast.style.cssText = `
        position: fixed;           /* fixed position in viewport */
        top: 20px;                /* 20px from the top edge */
        right: 20px;              /* 20px from the right edge */
        background: #ef4444;      /* red error background */
        color: white;             /* white foreground text */
        padding: 1rem 1.5rem;     /* internal padding */
        border-radius: 12px;      /* rounded corners */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); /* shadow */
        z-index: 1000;            /* ensure toast renders above other elements */
        animation: slideInRight 0.3s ease-out; /* slide-in entrance animation */
        max-width: 400px;         /* maximum toast width */
        font-weight: 500;         /* medium font weight */
    `;
    toast.textContent = message; // set the notification message text

    // Inject the slide-in keyframe animation dynamically
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%); /* start off-screen to the right */
                opacity: 0;
            }
            to {
                transform: translateX(0);    /* end at natural on-screen position */
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style); // inject animation styles into <head>

    document.body.appendChild(toast); // append toast to <body>

    // Auto-remove the toast and its styles after 5 seconds
    setTimeout(() => {
        toast.remove();  // remove toast from DOM
        style.remove();  // remove injected animation styles
    }, 5000);
}

// Verifies that the model API is reachable and responding
// Performs a health check on page load to detect connectivity issues early
async function checkAPIConnection() {
    try {
        // Simple GET request to the API health endpoint
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            console.log('✅ API connection established'); // confirmation in console
        } else {
            throw new Error('API health check failed');
        }
    } catch (error) {
        // Notify the user of connectivity issues
        console.warn('⚠️ Could not connect to API:', error.message);
        showError('Could not connect to the server. Make sure the API is running on port 8000');
    }
}

// Initializes the animated particle system rendered behind the UI
// Creates an attractive visual effect without interfering with functionality
function initParticles() {
    const canvas = document.getElementById('particles');   // get the drawing canvas element
    const ctx = canvas.getContext('2d');                  // 2D rendering context
    
    // Implementation note.
    function resizeCanvas() {
        canvas.width = window.innerWidth;   // match canvas width to viewport width
        canvas.height = window.innerHeight; // match canvas height to viewport height
    }
    
    resizeCanvas(); // initialize canvas dimensions on load
    window.addEventListener('resize', resizeCanvas); // keep canvas synced on window resize

    // Particle system configuration
    const particles = [];           // array holding all active particle instances
    const particleCount = 50;       // total number of particles

    // Initialize particles with randomized properties
    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,          // random initial x position
            y: Math.random() * canvas.height,         // random initial y position
            size: Math.random() * 3 + 1,              // random radius between 1 and 4px
            speedX: (Math.random() - 0.5) * 0.5,      // random horizontal velocity
            speedY: (Math.random() - 0.5) * 0.5,      // random vertical velocity
            opacity: Math.random() * 0.5 + 0.2        // opacity randomly between 0.2 and 0.7
        });
    }

    // Animation loop executed continuously via requestAnimationFrame
    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height); // clear the previous frame

        // Update position and draw each particle
        particles.forEach(particle => {
            // Update position based on current velocity vector
            particle.x += particle.speedX;
            particle.y += particle.speedY;

            // Bounce off screen edges by reversing velocity component
            if (particle.x < 0 || particle.x > canvas.width) particle.speedX *= -1;
            if (particle.y < 0 || particle.y > canvas.height) particle.speedY *= -1;

            // Draw particle as a semi-transparent white circle
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${particle.opacity})`;
            ctx.fill();
        });

        requestAnimationFrame(animateParticles); // schedule the next animation frame
    }

    animateParticles(); // start the animation loop
}

// Converts a byte count to a human-readable size string
// Implementation note.
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';  // special case: empty file
    
    const k = 1024;                     // binary conversion factor (1 KiB = 1024 bytes)
    const sizes = ['Bytes', 'KB', 'MB', 'GB']; // supported size unit labels
    
    // Implementation note.
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    // Format the value to 2 decimal places and append the appropriate unit label
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Retrieves detailed model metadata from the API
// Useful for debugging and verifying the active model configuration
async function getModelInfo() {
    try {
        // GET request to the model info endpoint
        const response = await fetch(`${API_BASE_URL}/model-info`);
        
        if (response.ok) {
            const info = await response.json();
            console.log('📊 Model information:', info); // shows in console
            return info; // return metadata for downstream use
        }
    } catch (error) {
        console.warn('Could not retrieve model information:', error);
    }
    return null; // return null on failure
}