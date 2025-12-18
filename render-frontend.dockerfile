# ---------- BUILDER ----------
FROM node:18-alpine AS builder
WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# ---------- RUNNER ----------
FROM node:18-alpine
WORKDIR /app

ENV NODE_ENV=production

COPY frontend/package*.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/next.config.js ./next.config.js

EXPOSE 3000

CMD ["npm", "start"]
