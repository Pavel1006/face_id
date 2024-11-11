import { useRef, useState } from "react";
import Webcam from "react-webcam";
import "./style.css";

const CameraCapture = () => {
  const webcamRef = useRef(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [loading, setLoading] = useState(false);
  const [recognizedName, setRecognizedName] = useState("");

  const capturePhoto = () => {
    const image = webcamRef.current.getScreenshot();
    setImageSrc(image);
    console.log("img was taken");
  };

  const handleSubmit = async () => {
    if (imageSrc) {
      setLoading(true);

      try {
        // Create a FormData object to send the image file to the backend
        const formData = new FormData();
        const file = dataURLtoFile(imageSrc, "captured.jpg");
        formData.append("file", file);
        console.log(file);

        // Make an API request to the FastAPI endpoint
        const response = await fetch("http://localhost:8000/upload/", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        console.log(data);

        // If face is recognized, update the recognized name
        if (data.recognized_name) {
          setRecognizedName(data.recognized_name);
        } else {
          setRecognizedName("No face detected");
        }
      } catch (error) {
        console.error("Error uploading image:", error);
        setRecognizedName("Error uploading image");
      } finally {
        setLoading(false);
      }
    }
  };

  // Helper function to convert base64 image to file
  const dataURLtoFile = (dataUrl, filename) => {
    let arr = dataUrl.split(","),
      mime = arr[0].match(/:(.*?);/)[1],
      bstr = atob(arr[1]),
      n = bstr.length,
      u8arr = new Uint8Array(n);

    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }

    return new File([u8arr], filename, { type: mime });
  };

  return (
    <div className="camera-capture">
      {!imageSrc ? (
        <div className="camera-view">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="webcam"
          />
          <button onClick={capturePhoto} className="capture-button">
            Capture Photo
          </button>
        </div>
      ) : (
        <div className="photo-preview">
          <img src={imageSrc} alt="Captured" className="captured-image" />
          <button onClick={() => setImageSrc(null)} className="retake-button">
            Retake Photo
          </button>
        </div>
      )}

      {imageSrc && !loading && (
        <div className="action-buttons">
          <button onClick={handleSubmit} className="submit-button">
            Submit for Recognition
          </button>
        </div>
      )}

      {loading && <p>Loading...</p>}
      {recognizedName && <p className="recognized-name">{recognizedName}</p>}
    </div>
  );
};

export default CameraCapture;
