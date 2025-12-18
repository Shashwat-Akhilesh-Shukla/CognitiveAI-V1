FROM node:18-alpine AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --frozen-lockfile

COPY frontend/ .
RUN npm run build


FROM node:18-alpine

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --frozen-lockfile

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/pages ./pages
COPY --from=builder /app/components ./components
COPY --from=builder /app/styles ./styles
COPY --from=builder /app/assets ./assets
COPY --from=builder /app/next.config.js ./next.config.js

ENV NODE_ENV=production
ENV NEXT_PUBLIC_API_URL=http://backend-service:8000

EXPOSE 3000

CMD ["npm", "run", "start"]
