import useState from "react";
import axios from "axios";

const API_URL = process.env.REACT_APP_API_URL || "https://backend-python-ed7p.onrender.com";

export default function ViolenceDetector() {
  const [imageResult, setImageResult] = useState(null);
  const [videoResult, setVideoResult] = useState(null);

  const uploadImage = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_URL}/detect-image`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setImageResult(res.data);
    } catch (err) {
      alert("Lỗi khi gửi ảnh!");
      console.error(err);
    }
  };

  const uploadVideo = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_URL}/detect-video`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setVideoResult(res.data);
    } catch (err) {
      alert("Lỗi khi gửi video!");
      console.error(err);
    }
  };

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Nhận diện bạo lực từ Ảnh / Video</h2>

      {/* Upload Ảnh */}
      <div className="mb-6">
        <h3 className="font-semibold mb-2">Tải ảnh lên</h3>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => uploadImage(e.target.files[0])}
        />

        {imageResult && (
          <div className="mt-3 p-3 bg-gray-100 rounded">
            <p><b>Xác suất bạo lực:</b> {imageResult.prob_violent?.toFixed(4)}</p>
            <p><b>Xác suất không bạo lực:</b> {imageResult.prob_nonviolent?.toFixed(4)}</p>
            <p><b>Kết luận:</b> {imageResult.violent ? "Có bạo lực" : "Không bạo lực"}</p>
          </div>
        )}
      </div>

      {/* Upload Video */}
      <div className="mb-6">
        <h3 className="font-semibold mb-2">Tải video lên</h3>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => uploadVideo(e.target.files[0])}
        />

        {videoResult && (
          <div className="mt-3 p-3 bg-gray-100 rounded">
            <p><b>Tỷ lệ bạo lực:</b> {(videoResult.violent_rate * 100).toFixed(2)}%</p>
            <p><b>Kết luận:</b> {videoResult.violent ? "Video có bạo lực" : "Video an toàn"}</p>
          </div>
        )}
      </div>
    </div>
  );
}
