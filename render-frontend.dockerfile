# ---------- BUILDER ----------
FROM node:18-alpine AS builder
WORKDIR /app

ENV NEXT_PUBLIC_API_URL=https://cognitiveai-v1.onrender.com
ENV NEXT_PUBLIC_BACKEND_URL=https://cognitiveai-v1.onrender.com
# Install deps
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build
COPY frontend/ .
RUN npm run build

# ---------- RUNNER ----------
FROM node:18-alpine
WORKDIR /app

ENV NODE_ENV=production

# Copy only what is needed to run
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/next.config.js ./next.config.js

EXPOSE 3000

CMD ["npm", "start"]
