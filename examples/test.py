from Galvatron import GalvatronUltra


galvatronMega = GalvatronUltra(use_4bit_quantization=True)
modality_data = {
    "Text": "Text data",
    "Image": "examples/100-trillion.png",
    # "Audio": "/path/to/audio.mp3",
    # "Video": "/path/to/video.mp4",  # Uncomment if video data is available
    # "Point Cloud": "/path/to/pointcloud.data",  # Uncomment if point cloud data is available
}
response = galvatronMega.generate(modality_data)
print(response)