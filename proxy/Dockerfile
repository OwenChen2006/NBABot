# Stage 1: build Angular
FROM node:18-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Caddy web + TLS + reverse proxy
FROM caddy:2-alpine
WORKDIR /srv
COPY --from=build /app/dist/app/ ./frontend/
COPY proxy/Caddyfile /etc/caddy/Caddyfile

