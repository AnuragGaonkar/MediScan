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
    setCompleted(false);
  };

  const handlePredict = async () => {
    if (!image || loading || completed) return;

    setLoading(true);
    setPrediction("");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
        // 🔥 FIXED: Essential headers for Render.com + file upload
        cache: "no-store",
      });

      // 🔥 FIXED: Robust error handling
      if (!res.ok) {
        const errorText = await res.text();
        console.error("Server response:", errorText);
        
        let errorData = {};
        try {
          errorData = JSON.parse(errorText);
        } catch {
          console.error("JSON parse failed:", errorText);
        }
        
        if (errorData.image_type === "Non-medical" || errorData.error === "Not a medical image") {
          setPrediction(`⚠️ ${errorData.disease || errorData.message || "Not a medical image"}\n\nPlease upload proper CT/MRI/X-Ray scans for accurate analysis.`);
        } else {
          setPrediction(`Server error: ${errorData.error || errorData.disease || 'Unknown error. Check console.'}`);
        }
        setCompleted(false);
        setLoading(false);
        return;
      }

      const data = await res.json();
      setPrediction(formatPrediction(data));
      setCompleted(true);
    } catch (err) {
      console.error("Prediction Error:", err);
      setPrediction("Network error. Please check your connection and try again.");
      setCompleted(false);
    } finally {
      setLoading(false);
    }
  };

  const formatPrediction = (data) => {
    let result = `IMAGE ANALYSIS REPORT\n\n`;
    result += `Image Type: ${data.image_type}\n`;
    result += `Type Confidence: ${data.image_type_confidence || 'N/A'}%\n\n`;
    result += `Status: ${data.status}\n`;
    result += `Disease Diagnosed: ${data.disease}\n`;
    result += `Prediction Confidence: ${data.disease_confidence || 'N/A'}%\n\n`;
    
    if (data.model_accuracy) {
      result += `Model Validation Accuracy: ${data.model_accuracy}%`;
    }
    
    return result;
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
        disabled={!image || loading || completed}
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
