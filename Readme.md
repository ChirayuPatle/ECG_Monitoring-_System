# Endpoint: `POST /api/ecg`
# Output Json
{
  "device_id": "ecg_esp8266_01",
  "sampling_rate": 250,
  "samples": [512, 515, 519, 521, 518]
}

# Steps to run server
- Start the server `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
- Endpoint `http://127.0.0.1:8000`
- Test the sample
`{
  "device_id": "test_device",
  "sampling_rate": 250,
  "samples": [
    512,
    518,
    525,
    531,
    527,
    520,
    515,
    510,
    516,
    523
  ]
}`

- Check response: 
`{
  "status": "success",
  "message": "ECG data received successfully",
  "device_id": "test_device",
  "samples_received": 10
}`

- 