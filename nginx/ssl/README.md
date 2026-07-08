Place your SSL certificate and key here.

For development (self-signed):
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
    -subj "/CN=localhost"

For production: replace with a CA-signed certificate.
