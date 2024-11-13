import React, { useState } from 'react';
import { Upload, X, Link, CheckCircle2, AlertCircle } from 'lucide-react';

interface TradeData {
  Date: string | null;
  Time: string | null;
  Ticker: string | null;
  Expiry: string | null;
  Strike: string | null;
  Instrument: string | null;
  Quantity: number;
  Net_proceeds: number;
  Symbol: string | null;
  DateTime: string | null;
  Proceeds: number;
  Comm_fee: number;
}

const FileUploadComponent = () => {
  const [file, setFile] = useState<File | null>(null);
  const [tradeData, setTradeData] = useState<TradeData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.size > 3 * 1024 * 1024) { // 3MB limit
        setError('File size exceeds 3MB limit');
        setSuccess(null);
        return;
      }
      setFile(selectedFile);
      handleUpload(selectedFile);
    }
  };

 const handleUpload = async (selectedFile: File) => {
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/parse-trades/`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setTradeData(data.slice(0, 20));
        setSuccess('File processed successfully');
        console.log('Upload successful:', data);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Error:', error);
      setError(error instanceof Error ? error.message : 'Failed to process file');
      setTradeData([]);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4">
      <div className="bg-gradient-to-b from-zinc-900 to-black rounded-2xl shadow-xl max-w-5xl w-full p-6 relative
                    border border-zinc-800/50 backdrop-blur-sm
                    shadow-[0_0_15px_rgba(0,0,0,0.7)]">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-2xl text-gray-200 font-light">Upload Trade Data</h2>
            <p className="text-sm text-gray-400 mt-2">
              Upload your trading data file in CSV or Excel format.
            </p>
          </div>
          <button className="text-gray-400 hover:text-gray-300">
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Notifications */}
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center gap-2 text-red-500">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}
        {success && (
          <div className="mb-4 p-3 rounded-lg bg-green-500/10 border border-green-500/20 flex items-center gap-2 text-green-500">
            <CheckCircle2 className="w-5 h-5" />
            <span>{success}</span>
          </div>
        )}

        {/* Upload Area */}
        <div className="mt-6">
          <div 
            className={`border-2 border-dashed rounded-xl p-8 
                       cursor-pointer text-center backdrop-blur-sm
                       transition-colors duration-200
                       ${error ? 'border-red-500/50 bg-red-500/5' : 
                       success ? 'border-green-500/50 bg-green-500/5' : 
                       'border-gray-600 bg-black/40 hover:bg-black/60'}`}
          >
            <input
              type="file"
              onChange={handleFileChange}
              accept=".csv,.xlsx,.xls"
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className={`w-10 h-10 mx-auto mb-4 ${
                error ? 'text-red-500' : 
                success ? 'text-green-500' : 
                'text-gray-400'
              }`} />
              <div className="text-gray-300">
                Drag & Drop or <span className="text-blue-400">Choose file</span>
              </div>
              <p className="text-sm text-gray-500 mt-2">
                CSV or Excel files up to 3MB
              </p>
            </label>
          </div>

          {/* File Preview */}
          {file && (
            <div className="mt-4 p-3 bg-black/40 rounded-lg flex justify-between items-center backdrop-blur-sm">
              <div className="flex items-center">
                <div className="text-gray-300">{file.name}</div>
                <div className="text-gray-500 text-sm ml-2">
                  ({(file.size / 1024 / 1024).toFixed(2)}MB)
                </div>
              </div>
              <button 
                onClick={() => {
                  setFile(null);
                  setError(null);
                  setSuccess(null);
                  setTradeData([]);
                }} 
                className="text-gray-400 hover:text-gray-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Table Results */}
          {tradeData.length > 0 && (
            <div className="mt-6">
              <div className="max-h-[500px] overflow-auto rounded-lg bg-black/40 backdrop-blur-sm">
                <table className="w-full border-collapse min-w-max">
                  <thead className="sticky top-0 bg-black/60 backdrop-blur-sm">
                    <tr className="border-b border-zinc-800">
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Date</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Time</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Ticker</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Expiry</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Strike</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Instrument</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Quantity</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Net Proceeds</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Proceeds</th>
                      <th className="px-6 py-3 text-left text-sm text-gray-300 font-medium">Comm/Fee</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/50">
                    {tradeData.map((trade, index) => (
                      <tr key={index} className="hover:bg-zinc-800/30 transition-colors">
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Date}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Time}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Ticker}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Expiry}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Strike}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Instrument}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Quantity?.toFixed(2)}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Net_proceeds?.toFixed(2)}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Proceeds?.toFixed(2)}</td>
                        <td className="px-6 py-3 text-sm text-gray-300 whitespace-nowrap">{trade.Comm_fee?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUploadComponent;
