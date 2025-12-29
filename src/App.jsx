import React, { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [completed, setCompleted] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL;

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setPrediction("");
    setCompleted(false); // ✅ Resets on new image
  };

  const handlePredict = async () => {
    if (!image || loading || completed) return; // ✅ Prevents re-click

    setLoading(true);
    setPrediction("");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text);
      }

      const data = await res.json();
      setPrediction(formatPrediction(data));
      setCompleted(true);
    } catch (err) {
      console.error("Prediction Error:", err);
      setPrediction("Server error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const formatPrediction = (data) => {
    return (
      `Image Type: ${data.image_type}\n` +
      `Status: ${data.status}\n` +
      `Disease Diagnosed: ${data.disease}`
    );
  };

  return (
    <div className={`app-container ${preview ? "has-preview" : "no-preview"}`}>
      <header className="app-header">
        <h1>MediScan</h1>
        <p>Medical Image Disease Classification</p>
      </header>

      <div className="upload-section">
        <label className="file-label">
          <span>Upload Medical Image</span>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input"
          />
        </label>
      </div>

      {preview && (
        <div className="report-section">
          <h2 className="report-title">Scan Report</h2>
          <div className="report-grid">
            <div className="report-card image-card">
              <h3>Uploaded Image</h3>
              <div className="image-container">
                <img src={preview} alt="Medical scan preview" />
              </div>
            </div>

            <div className="report-card details-card">
              <h3>Image Details</h3>
              <div className="detail-item">
                <span className="detail-label">File:</span>
                <span className="detail-value">{image?.name}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Type:</span>
                <span className="detail-value">{image?.type}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Size:</span>
                <span className="detail-value">{(image?.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            </div>

            <div className="report-card prediction-card">
              <h3>AI Prediction</h3>
              <div className="prediction-container">
                <pre className="prediction-text">
                  {prediction || "No prediction yet..."}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}

      <button
        className={`predict-button ${loading ? "loading" : ""} ${completed ? "completed" : ""}`}
        onClick={handlePredict}
        disabled={!image || loading || completed} // ✅ DISABLED after completion
      >
        {loading ? (
          <>
            <span className="spinner"></span>
            Analyzing Scan...
          </>
        ) : completed ? (
          <>
            <span className="checkmark">✓</span>
            Analysis Complete
          </>
        ) : (
          "Run Analysis"
        )}
      </button>

      {loading && <div className="loading-overlay">Processing your medical image...</div>}
    </div>
  );
}

export default App;
