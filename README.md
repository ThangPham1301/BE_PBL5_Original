# PBL5 Backend

REST API cho hệ thống chấm công, quản lý nghỉ phép, ca làm việc và nhận diện khuôn mặt.

## Chạy dự án

```bash
cd attendance_system
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Cấu trúc

```text
attendance_system/
├── apps/                # Các Django app theo nghiệp vụ
│   ├── accounts/
│   ├── attendance/
│   ├── employees/
│   ├── face_recognition/
│   ├── leaves/
│   ├── reports/
│   └── shifts/
├── config/              # Settings, URL, ASGI, WSGI và Celery
├── scripts/             # Công cụ chẩn đoán/chạy thủ công
├── manage.py
└── requirements.txt
```

Không đổi tên các thư mục trong `apps` sau khi đã tạo migration. Logic nghiệp vụ nên nằm trong app sở hữu dữ liệu tương ứng; script thử nghiệm không đặt cạnh `manage.py`.

## Kiểm tra

```bash
python manage.py check
python manage.py test
python manage.py makemigrations --check --dry-run
```
