FROM nginx:1.23.4-alpine

RUN apk --no-cache add curl

RUN mkdir -p /visualizer \
    && curl -SL https://tenka1.klab.jp/2024/visualizer.tar.gz \
    | tar -xzC /visualizer

COPY portal /portal
COPY nginx.conf /etc/nginx/nginx.conf
COPY mime.types /etc/nginx/mime.types
