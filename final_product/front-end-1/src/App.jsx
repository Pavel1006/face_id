import { useState } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [name, setName] = useState("");
  const [responseMessage, setResponseMessage] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleNameChange = (event) => {
    setName(event.target.value);
  };

  const handleRegister = async () => {
    if (!selectedFile || !name) {
      setResponseMessage("Please enter a name and select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/register/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResponseMessage(response.data.message || response.data.error);
    } catch (error) {
      setResponseMessage(
        "Error: " + (error.response?.data?.error || "Something went wrong")
      );
    }
  };

  const handleRecognize = async () => {
    if (!selectedFile) {
      setResponseMessage("Please select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/recognize/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResponseMessage(response.data.message || response.data.error);
    } catch (error) {
      setResponseMessage(
        "Error: " + (error.response?.data?.error || "Something went wrong")
      );
    }
  };

  return (
    <div
      style={{ textAlign: "center", marginTop: "50px", marginLeft: "500px" }}
    >
      <h2>Face Recognition App</h2>
      <input
        type="file"
        onChange={handleFileChange}
        style={{ marginRight: "auto", marginLeft: "auto" }}
      />
      <br />
      <input
        type="text"
        placeholder="Enter name for registration"
        value={name}
        onChange={handleNameChange}
        style={{ marginTop: "10px", padding: "5px" }}
      />
      <br />
      <button
        onClick={handleRegister}
        style={{ margin: "10px", padding: "10px" }}
      >
        Register
      </button>
      <button onClick={handleRecognize} style={{ padding: "10px" }}>
        Recognize
      </button>
      <p>{responseMessage}</p>
    </div>
  );
}

export default App;
