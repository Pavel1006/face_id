import { useRef, useState } from "react";
import Webcam from "react-webcam";
import "./style.css";

const CameraCapture = () => {
  const webcamRef = useRef(null);
  const [imageSrc, setImageSrc] = useState(null);

  const capturePhoto = () => {
    const image = webcamRef.current.getScreenshot();
    setImageSrc(image);
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
    </div>
  );
};

export default CameraCapture;
