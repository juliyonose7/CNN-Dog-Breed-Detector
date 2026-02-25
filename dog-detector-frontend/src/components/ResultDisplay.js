import React from 'react';
import './ResultDisplay.css';

const ResultDisplay = ({ prediction }) => {
  if (prediction.error) {
    return (
      <div className="result-card glass error glow-red">
        <div className="result-icon">❌</div>
        <h3>Error</h3>
        <p>{prediction.error}</p>
      </div>
    );
  }

  // Nuevo formato para API de 119 clases
  const isSuccess = prediction.success;
  const isDog = prediction.is_dog;
  const topPrediction = prediction.top_predictions?.[0];
  
  if (!isSuccess || !isDog || !topPrediction) {
    return (
      <div className="result-card glass glow-blue">
        <div className="result-icon">❌</div>
        <h3>No es un perro</h3>
        <p>No se detectó un perro en esta imagen.</p>
      </div>
    );
  }

  const confidence = (topPrediction.confidence * 100).toFixed(1);
  const isConfident = topPrediction.is_confident;
  const glowClass = isConfident ? 'glow-green' : 'glow-yellow';

  return (
    <div className={`result-card glass ${glowClass}`}>
      <div className="result-header">
        <div className="result-icon">🐕</div>
        <h3 className="result-title">
          ¡Es un {topPrediction.breed}!
        </h3>
      </div>
      
      <div className="result-details">
        <div className="confidence-container">
          <div className="confidence-label">Confianza del modelo:</div>
          <div className="confidence-bar-container">
            <div 
              className={`confidence-bar ${isConfident ? 'success' : 'warning'}`}
              style={{ width: `${confidence}%` }}
            ></div>
          </div>
          <div className="confidence-value">{confidence}%</div>
        </div>
        
        <div className="prediction-details">
          <div className="detail-item">
            <span className="detail-label">Raza más probable:</span>
            <span className={`detail-value ${isConfident ? 'positive' : 'warning'}`}>
              {topPrediction.breed}
            </span>
          </div>
          
          <div className="detail-item">
            <span className="detail-label">Precisión del modelo:</span>
            <span className="detail-value">ResNet50 - 119 clases</span>
          </div>
          
          <div className="detail-item">
            <span className="detail-label">Tiempo de procesamiento:</span>
            <span className="detail-value">{prediction.processing_time}s</span>
          </div>
        </div>

        {/* Top 5 predicciones */}
        <div className="top-predictions">
          <div className="top-predictions-title">Top 5 Predicciones:</div>
          {prediction.top_predictions?.slice(0, 5).map((pred, index) => (
            <div key={index} className="prediction-item">
              <span className="prediction-rank">#{index + 1}</span>
              <span className="prediction-breed">{pred.breed}</span>
              <span className="prediction-confidence">{(pred.confidence * 100).toFixed(1)}%</span>
              {pred.is_confident && <span className="confidence-badge">✓</span>}
            </div>
          ))}
        </div>
      </div>
      
      <div className="result-description">
        {isConfident ? (
          <p>🎉 ¡Excelente! Tengo alta confianza en que esta es la raza correcta.</p>
        ) : (
          <p>🤔 Esta es mi mejor estimación, pero podría beneficiarse de una imagen más clara.</p>
        )}
      </div>
    </div>
  );
};

export default ResultDisplay;