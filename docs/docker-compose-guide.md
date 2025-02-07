# Docker Compose Guide

## 📌 Introduction
Docker Compose is a tool used to define and manage multi-container Docker applications. It allows you to configure services, networks, and volumes in a `docker-compose.yml` file and run them with a single command.

---

## 🛠️ Installing Docker Compose

### 🔹 macOS (via Homebrew)
```sh
brew install docker-compose
```

### 🔹 Linux
```sh
sudo apt update && sudo apt install docker-compose -y
```

### 🔹 Windows
Download and install Docker Desktop, which includes Docker Compose:
[Docker Install for Windows](https://docs.docker.com/desktop/install/windows-install/)

### 🔹 Verify Installation
```sh
docker-compose --version
```

---

## 📄 Writing a `docker-compose.yml` File
A basic example of a `docker-compose.yml` file:
```yaml
version: '3.8'
services:
  web:
    image: nginx:alpine
    ports:
      - "8080:80"
    networks:
      - my_network
  
db:
    image: postgres:latest
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - db_data:/var/lib/postgresql/data
    networks:
      - my_network

networks:
  my_network:

volumes:
  db_data:
```

### 🔹 Explanation of Commands
- `version` : Specifies the Docker Compose file format version.
- `services` : Defines containers that make up the application.
- `image` : Specifies the Docker image to use for the service.
- `ports` : Maps container ports to host ports (`host:container`).
- `environment` : Defines environment variables.
- `volumes` : Maps data volumes for persistent storage.
- `networks` : Defines networks for inter-service communication.

---

## 🚀 Running and Managing Docker Compose

### 🔹 Start Services
```sh
docker-compose up -d
```
- `-d` : Runs in detached mode (background).

### 🔹 Stop and Remove Services
```sh
docker-compose down
```
- Stops and removes containers, networks, and volumes (unless named volumes are used).

### 🔹 View Running Services
```sh
docker-compose ps
```
- Lists all active services.

### 🔹 View Logs
```sh
docker-compose logs -f
```
- `-f` : Follows real-time logs.

### 🔹 Restart Services
```sh
docker-compose restart
```
- Restarts all services.

### 🔹 Execute a Command in a Running Container
```sh
docker-compose exec web ls /usr/share/nginx/html
```
- Runs `ls` inside the `web` container.

### 🔹 Rebuild Services with Changes
```sh
docker-compose up --build -d
```
- Rebuilds images before starting containers.

---

## 🎯 Conclusion
This guide provides a structured approach to using Docker Compose for multi-container applications. Use it as a quick reference for setting up and managing your containers efficiently! 🚀
