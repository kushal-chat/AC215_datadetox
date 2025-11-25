FROM node:20-slim
WORKDIR /app

# Accept build argument
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . ./

# Expose port
EXPOSE 3000

# Run in development mode with hot reload
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]