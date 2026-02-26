import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import ParticleBackground from './components/ParticleBackground';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handleImageUpload = async (file) => {
    setLoading(true);
    setPrediction(null);
    
    // Store the uploaded image for display
    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction request failed');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Error:', error);
      setPrediction({
        error: 'Error processing image. Make sure the server is running.'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setUploadedImage(null);
    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage);
    }
  };

  return (
    <div className="App">
      <ParticleBackground />
      
      <div className="container">
        <header className="header">
          <h1>🐕 Dog Detector AI</h1>
          <p className="subtitle">
            Upload an image and discover if there is a dog using our AI
          </p>
        </header>

        <main className="main-content">
          {!uploadedImage ? (
            <ImageUpload onImageUpload={handleImageUpload} loading={loading} />
          ) : (
            <div className="result-container">
              <div className="uploaded-image-container glass">
                <img 
                  src={uploadedImage} 
                  alt="Uploaded image" 
                  className="uploaded-image"
                />
              </div>
              
              {loading && (
                <div className="loading-container glass">
                  <div className="spinner"></div>
                  <p>Analyzing image...</p>
                </div>
              )}
              
              {prediction && !loading && (
                <ResultDisplay prediction={prediction} />
              )}
              
              <button 
                className="btn btn-reset" 
                onClick={handleReset}
                disabled={loading}
              >
                🔄 Upload another image
              </button>
            </div>
          )}
        </main>

        <footer className="footer">
          <p>Powered by PyTorch &amp; FastAPI | Optimized for AMD 7900XTX</p>
        </footer>
      </div>
    </div>
  );
}

export default App;