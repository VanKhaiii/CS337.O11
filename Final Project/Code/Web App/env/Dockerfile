FROM ubuntu:latest

RUN apt-get update -y

RUN apt-get install -y python3-pip python3-dev build-essential hdf5-tools libgl1 libgtk2.0-dev
    
# Thiết lập thư mục làm việc
WORKDIR /music_app

# Sao chép các tệp cần thiết vào thư mục làm việc
COPY . /music_app

# Cài đặt các gói phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng cho dịch vụ
EXPOSE 8080

# # Khởi động ứng dụng
CMD ["python", "app.py"]