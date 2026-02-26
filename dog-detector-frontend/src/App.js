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
        throw new Error('Error en la predicción');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Error:', error);
      setPrediction({
        error: 'Error al procesar la imagen. Asegúrate de que el servidor esté funcionando.'
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
            Sube una imagen y descubre si hay un perro con nuestra IA
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
                  alt="Imagen subida" 
                  className="uploaded-image"
                />
              </div>
              
              {loading && (
                <div className="loading-container glass">
                  <div className="spinner"></div>
                  <p>Analizando imagen...</p>
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
                🔄 Subir otra imagen
              </button>
            </div>
          )}
        </main>

        <footer className="footer">
          <p>Powered by PyTorch & FastAPI | Optimizada para AMD 7900XTX</p>
        </footer>
      </div>
    </div>
  );
}

export default App;