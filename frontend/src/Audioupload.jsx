// src/AudioUpload.js
import React from 'react';
import './Audioupload.css'; // Create this CSS file for styling

const AudioUpload = () => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log('Uploaded file:', file);
      // You can add further processing for the uploaded file here
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload Audio File</h2>
      <input
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="upload-input"
      />
    </div>
  );
};

export default AudioUpload;