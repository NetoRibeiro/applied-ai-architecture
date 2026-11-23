# Multi-stage Dockerfile for Applied AI Architecture Workbench
FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

FROM node:22-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

COPY package.json ./
RUN npm install --omit=dev

COPY --from=builder /app/dist ./dist
COPY server.ts tsconfig.json firebase-applet-config.json LOGBOOK.md ./
COPY src/types.ts ./src/types.ts
COPY 01-Phase-1-Math-and-ML-Core ./01-Phase-1-Math-and-ML-Core
COPY 02-Phase-2-Deep-Learning ./02-Phase-2-Deep-Learning
COPY 03-Phase-3-Advanced-RAG ./03-Phase-3-Advanced-RAG
COPY 04-Phase-4-Fine-Tuning ./04-Phase-4-Fine-Tuning
COPY 05-Phase-5-MLOps-and-Agents ./05-Phase-5-MLOps-and-Agents
COPY 06-Phase-6-Causal-Inference ./06-Phase-6-Causal-Inference
COPY 07-Phase-7-Governance-and-Scale ./07-Phase-7-Governance-and-Scale

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3000/api/health || exit 1

CMD ["./node_modules/.bin/tsx", "server.ts"]
