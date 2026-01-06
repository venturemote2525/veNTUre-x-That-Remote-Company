FROM node:20-slim

ENV EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0

WORKDIR /app

# Install expo-cli globally
RUN npm install -g expo-cli @expo/ngrok

# Copy package files first for better caching
COPY package.json package-lock.json* ./

RUN npm install --legacy-peer-deps

# Copy the rest of the app
COPY . .

EXPOSE 8081 19000 19001 19002

CMD ["npx", "expo", "start", "--tunnel"]
