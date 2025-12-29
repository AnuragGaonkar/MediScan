import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState('');
  const API_URL = import.meta.env.VITE_API_URL;
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPrediction('');
    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append('file', image);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      // Format the response prediction
      setPrediction(formatPrediction(data));
    } catch (err) {
      console.error('Prediction Error:', err);
      setPrediction('Server error. Please try again.');
    }
  };

  // Function to format the prediction in a readable way
  const formatPrediction = (data) => {
    return (
      `Image Type: ${data.image_type}\n` +
      `Status: ${data.status}\n` +
      `Disease Diagnosed: ${data.disease}`
    );
  };

  return (
    <div className="container">
      <div className="title">Medical Image Classifier</div>

      <div className="upload-section">
        <label className="file-label" htmlFor="fileInput">Upload Medical Image</label>
        <input
          id="fileInput"
          className="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        />
      </div>

      {preview && (
        <div className="horizontal-report">
          <h2>Scan Report</h2>
          <hr />
          <div className="report-content">
            <div className="report-section image-section">
              <h3>Uploaded Image</h3>
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
            <div className="report-section details-section">
              <h3>Details</h3>
              <p>Resolution: {preview ? 'Auto-detected' : '—'}</p>
              <p>File Type: {image?.type || '—'}</p>
              <p>File Name: {image?.name || '—'}</p>
            </div>
            <div className="report-section findings-section">
              <h3>Prediction</h3>
              <pre className="prediction">{prediction || 'No prediction yet'}</pre>
            </div>
          </div>
        </div>
      )}

      <button className="predict-btn" onClick={handlePredict}>Predict</button>
    </div>
  );
}

export default App;
  