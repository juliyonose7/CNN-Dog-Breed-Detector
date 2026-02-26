import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import './ImageUpload.css';

const ImageUpload = ({ onImageUpload, loading }) => {
  const [isDragActive, setIsDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      onImageUpload(file);
    }
  }, [onImageUpload]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    multiple: false,
    disabled: loading,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
  });

  return (
    <div 
      {...getRootProps()} 
      className={`upload-zone glass ${isDragActive ? 'drag-active glow-blue' : ''} ${loading ? 'loading' : ''}`}
    >
      <input {...getInputProps()} />
      
      <div className="upload-content">
        {loading ? (
          <>
            <div className="spinner"></div>
            <p>Processing image...</p>
          </>
        ) : (
          <>
            <div className="upload-icon">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path 
                  d="M12 15L12 2M12 2L8 6M12 2L16 6M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            
            <h3>Upload an image</h3>
            
            <p className="upload-description">
              Drag and drop an image here, or click to select
            </p>
            
            <div className="upload-formats">
              <span>Supported formats: JPG, PNG, GIF, BMP, WebP</span>
            </div>
            
            <button type="button" className="btn upload-btn">
              📁 Select file
            </button>
          </>
        )}
      </div>
      
      <div className="upload-background-pattern"></div>
    </div>
  );
};

export default ImageUpload;