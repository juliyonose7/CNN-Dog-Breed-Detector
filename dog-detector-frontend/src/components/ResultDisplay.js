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

  // New format for API of 119 classes
  const isSuccess = prediction.success;
  const isDog = prediction.is_dog;
  const topPrediction = prediction.top_predictions?.[0];
  
  if (!isSuccess || !isDog || !topPrediction) {
    return (
      <div className="result-card glass glow-blue">
        <div className="result-icon">❌</div>
        <h3>Not a dog</h3>
        <p>No dog was detected in this image.</p>
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
          🐕 {topPrediction.breed}!
        </h3>
      </div>
      
      <div className="result-details">
        <div className="confidence-container">
          <div className="confidence-label">Model confidence:</div>
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
            <span className="detail-label">Most likely breed:</span>
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

        {/* Top 5 predictions */}
        <div className="top-predictions">
          <div className="top-predictions-title">Top 5 Predictions:</div>
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
          <p>🎉 High confidence in this breed identification.</p>
        ) : (
          <p>🤔 Best estimate — a clearer image may improve accuracy.</p>
        )}
      </div>
    </div>
  );
};

export default ResultDisplay;